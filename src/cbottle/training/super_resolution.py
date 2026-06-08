# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
from typing import Optional

import earth2grid
import torch
import torch.distributed as dist
import einops

import cbottle.checkpointing
import cbottle.config.environment as config
import cbottle.models
from cbottle import healpix_utils
from cbottle.datasets import samplers
from cbottle.datasets.dataset_2d import HealpixDatasetV5
from cbottle import distributed as cbottle_dist


class EDMLossSR:
    def __init__(
        self,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, img_clean, img_lr, pos_embed):
        labels = None
        rnd_normal = torch.randn([img_clean.shape[0], 1, 1, 1], device=img_clean.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(img_clean) * sigma
        sigma_lr = None
        D_yn = net(
            img_clean + n,
            sigma,
            class_labels=labels,
            condition=img_lr,
            position_embedding=pos_embed,
            augment_labels=sigma_lr,
        )
        loss = weight * ((D_yn - img_clean) ** 2)
        return loss


def load_checkpoint(path: str, *, network, optimizer, scheduler, map_location) -> int:
    with cbottle.checkpointing.Checkpoint(path) as checkpoint:
        if isinstance(network, torch.nn.parallel.DistributedDataParallel):
            checkpoint.read_model(net=network.module)
        else:
            checkpoint.read_model(net=network)

        with checkpoint.open("loop_state.pth", "r") as f:
            training_state = torch.load(f, weights_only=True, map_location=map_location)
            optimizer.load_state_dict(training_state["optimizer_state_dict"])
            scheduler.load_state_dict(training_state["scheduler_state_dict"])
            step = training_state["step"]

    return step


def save_checkpoint(path, *, model_config, network, optimizer, scheduler, step, loss):
    with cbottle.checkpointing.Checkpoint(path, "w") as checkpoint:
        if isinstance(network, torch.nn.parallel.DistributedDataParallel):
            checkpoint.write_model(network.module)
        else:
            checkpoint.write_model(network)
        checkpoint.write_model_config(model_config)

        with checkpoint.open("loop_state.pth", "w") as f:
            torch.save(
                {
                    "step": step,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": loss,
                },
                f,
            )


def find_latest_checkpoint(output_path: str) -> str:
    max_index_file = " "
    max_index = -1
    for filename in os.listdir(output_path):
        if filename.startswith("cBottle-SR-") and filename.endswith(".zip"):
            index_str = filename.split("-")[-1].split(".")[0]
            try:
                index = int(index_str)
                if index > max_index:
                    max_index = index
                    max_index_file = filename
            except ValueError:
                continue
    path = os.path.join(output_path, max_index_file)
    return path


class Mockdataset(torch.utils.data.Dataset):
    grid = earth2grid.healpix.Grid(
        level=10, pixel_order=earth2grid.healpix.PixelOrder.NEST
    )
    fields_out = HealpixDatasetV5.fields_out

    def __getitem__(self, i):
        npix = 12 * 4**self.grid.level
        return {"target": torch.randn(len(HealpixDatasetV5.fields_out), 1, npix)}

    def __len__(self):
        return 1


class BatchedPatchIterator:
    def __init__(
        self,
        net: torch.nn.Module,
        training_dataset_grid: earth2grid.healpix.Grid,
        lr_level: int,
        img_resolution: int,
        time_length: int = 1,
        padding: Optional[int] = None,
        shuffle: bool = True,
    ):
        self.net = net
        self.lr_level = lr_level
        self.time_length = time_length
        self.img_resolution = img_resolution
        self.sr_level = training_dataset_grid.level
        self.padding = padding if padding is not None else img_resolution // 2
        self.shuffle = shuffle

        # Setup regridders
        low_res_grid = earth2grid.healpix.Grid(
            lr_level, pixel_order=earth2grid.healpix.PixelOrder.NEST
        )
        lat = torch.linspace(-90, 90, 128)[:, None].cpu().numpy()
        lon = torch.linspace(0, 360, 128)[None, :].cpu().numpy()
        self.regrid_to_latlon = low_res_grid.get_bilinear_regridder_to(lat, lon).cuda()
        self.regrid = earth2grid.get_regridder(low_res_grid, training_dataset_grid)
        self.regrid.cuda().float()
        self.coordinate_map = self.make_coordinate_map(self.sr_level, self.padding)

    @staticmethod
    def make_coordinate_map(level: int, padding: int, device="cuda") -> torch.Tensor:
        """
        Returns a tensor of shape (1, 12 * X * Y), where X=Y=NSIDE with padding
        Pixel ID layout:
            id = face * X * Y + row * Y + col
        """
        nside = 2**level
        nside_padded = nside + 2 * padding
        ids = torch.arange(12 * nside_padded**2, dtype=torch.float32, device=device)
        ids = ids.view(1, 12, nside_padded, nside_padded)
        return ids

    def extract_positional_embeddings(self, patch_coord_map, padded_pe):
        # Decode the top-left ID of every patch to get its patch coordinates
        ids = patch_coord_map[:, 0, 0, 0].long()
        npix_padded = self.coordinate_map.shape[-1]
        face, rem = (
            torch.div(ids, npix_padded**2, rounding_mode="floor"),
            torch.remainder(ids, npix_padded**2),
        )
        row, col = (
            torch.div(rem, npix_padded, rounding_mode="floor"),
            torch.remainder(rem, npix_padded),
        )

        # get the positional embedding slice corresponding to each patch
        lpe = torch.stack(
            [
                padded_pe[
                    :,
                    int(f),
                    int(r) : int(r) + self.img_resolution,
                    int(c) : int(c) + self.img_resolution,
                ]
                for f, r, c in zip(face, row, col)
            ],
            dim=0,
        )
        return lpe

    def compute_low_res_conditioning(self, target):
        # Get low res version
        lr = target.clone()
        for _ in range(self.sr_level - self.lr_level):
            lr = healpix_utils.average_pool(lr)
        global_lr = self.regrid_to_latlon(lr.double())[None,].cuda()
        lr = self.regrid(lr)
        return lr, global_lr

    def __call__(self, batch, batch_size):
        target = batch["target"].cuda()
        target = einops.rearrange(target, "c t x -> (t c) x", t=self.time_length)

        lr, global_lr = self.compute_low_res_conditioning(target)

        # Create patches
        patches = healpix_utils.to_patches(
            [target, lr],
            patch_size=self.img_resolution,
            batch_size=batch_size,
            padding=self.padding,
            pre_padded_tensors=[self.coordinate_map],
            shuffle=self.shuffle,
        )
        del target, lr

        for ltarget, llr, patch_coord_map, _ in patches:
            faces_pe = healpix_utils.to_faces(self.net.module.model.pos_embed)
            padded_pe = earth2grid.healpix.pad(faces_pe, padding=self.padding)

            lpe = self.extract_positional_embeddings(patch_coord_map, padded_pe)

            global_lr_repeat = einops.repeat(
                global_lr,
                "1 (t c) x y -> (b t) c x y",
                b=llr.shape[0],
                t=self.time_length,
            )
            lpe = einops.repeat(lpe, "b c x y -> (b t) c x y", t=self.time_length)
            llr = einops.rearrange(
                llr.cuda(), "b (t c) x y -> (b t) c x y", t=self.time_length
            )
            ltarget = einops.rearrange(
                ltarget.cuda(), "b (t c) x y -> (b t) c x y", t=self.time_length
            )

            llr = torch.cat((llr, global_lr_repeat), dim=1)

            yield lpe, ltarget, llr


def train(
    output_path: str,
    customized_dataset=None,
    lr_level=6,
    train_batch_size=64,
    test_batch_size=64,
    valid_min_samples: int = 1,
    num_steps: int = int(4e7),
    log_freq: int = 1000,
    test_fast: bool = False,
    dataloader_num_workers: int = 3,
    bf16: bool = False,
):
    """
    Args:
        test_fast: used for rapid testing. E.g. uses mocked data to avoid
            network I/O.
    """
    cbottle_dist.init()

    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
    WORLD_SIZE = cbottle_dist.get_world_size()
    WORLD_RANK = cbottle_dist.get_rank()

    print(f"Rank {WORLD_RANK}/{WORLD_SIZE}; Local rank: {LOCAL_RANK}")

    os.makedirs(output_path, exist_ok=True)
    training_sampler = None
    test_sampler = None
    # dataloader
    if test_fast:
        training_dataset = Mockdataset()
        test_dataset = Mockdataset()
    elif customized_dataset:
        training_dataset = customized_dataset(
            split="train",
        )
        test_dataset = customized_dataset(
            split="test",
        )
    else:
        training_dataset = HealpixDatasetV5(
            path=config.RAW_DATA_URL,
            train=True,
            healpixpad_order=False,
            land_path=config.LAND_DATA_URL_10,
        )
        test_dataset = HealpixDatasetV5(
            path=config.RAW_DATA_URL,
            train=False,
            healpixpad_order=False,
            land_path=config.LAND_DATA_URL_10,
        )
        training_sampler = samplers.InfiniteSequentialSampler(training_dataset)
        valid_min_samples = max(valid_min_samples, WORLD_SIZE)
        test_sampler = samplers.distributed_split(
            samplers.subsample(
                test_dataset, min_samples=max(WORLD_SIZE, valid_min_samples)
            )
        )
    training_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=None,
        num_workers=dataloader_num_workers,
        sampler=training_sampler,
        pin_memory=True,
        multiprocessing_context="fork" if dataloader_num_workers > 0 else None,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=None,
        shuffle=None,
        sampler=test_sampler,
        pin_memory=True,
        num_workers=0,
    )

    loss_fn = EDMLossSR()
    out_channels = len(training_dataset.fields_out)

    # the model takes in both local and global lr channels
    local_lr_channels = out_channels
    global_lr_channels = out_channels
    model_config = cbottle.models.ModelConfigV1(
        architecture="unet_hpx1024_patch",
        condition_channels=local_lr_channels + global_lr_channels,
        out_channels=out_channels,
    )
    img_resolution = model_config.img_resolution
    model_config.level = training_dataset.grid.level
    net = cbottle.models.get_model(model_config)
    net.train().requires_grad_(True).cuda()
    net.cuda(LOCAL_RANK)
    net = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[LOCAL_RANK], find_unused_parameters=False
    )

    patch_iterator = BatchedPatchIterator(
        net,
        training_dataset.grid,
        lr_level,
        img_resolution,
    )

    # optimizer
    params = list(filter(lambda kv: "pos_embed" in kv[0], net.named_parameters()))
    base_params = list(
        filter(lambda kv: "pos_embed" not in kv[0], net.named_parameters())
    )
    params = [i[1] for i in params]
    base_params = [i[1] for i in base_params]
    optimizer = torch.optim.SGD(
        [{"params": base_params}, {"params": params, "lr": 5e-4}], lr=1e-7, momentum=0.9
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50000, gamma=0.6)
    tic = time.time()
    step = 0
    train_loss_list = []
    val_loss_list = []

    # load checkpoint
    path = find_latest_checkpoint(output_path)

    try:
        map_location = {
            "cuda:%d" % 0: "cuda:%d" % int(LOCAL_RANK)
        }  # map_location='cuda:{}'.format(self.params.local_rank)
        step = load_checkpoint(
            path,
            network=net,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=map_location,
        )
        step = step + 1
        print(f"Loaded network and optimizer states from {path}")
        if WORLD_RANK == 0:
            for p in optimizer.param_groups:
                print(p["lr"], p["initial_lr"])
    except FileNotFoundError:
        if WORLD_RANK == 0:
            print("Could not load network and optimizer states")

    # training loop
    old_pos = None
    old_pos2 = None
    old_conv = None
    old_conv2 = None
    running_loss = 0

    if WORLD_RANK == 0:
        print("training begin...", flush=True)

    while True:
        for batch in training_loader:
            for lpe, ltarget, llr in patch_iterator(batch, train_batch_size):
                step += 1
                optimizer.zero_grad()
                # Compute the loss and its gradients
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                    loss = loss_fn(net, img_clean=ltarget, img_lr=llr, pos_embed=lpe)
                loss = loss.sum()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1e6)
                optimizer.step()
                # avoid synchronizing gpu
                dist.all_reduce(loss)
                running_loss += loss.item()

                # logging
                if step % log_freq == 0:
                    with torch.no_grad():
                        val_running_loss = 0
                        for batch in test_loader:
                            count = 0
                            for lpe, ltarget, llr in patch_iterator(
                                batch, test_batch_size
                            ):
                                with torch.autocast(
                                    "cuda", dtype=torch.bfloat16, enabled=bf16
                                ):
                                    loss = loss_fn(
                                        net,
                                        img_clean=ltarget,
                                        img_lr=llr,
                                        pos_embed=lpe,
                                    )
                                loss = loss.sum()
                                dist.all_reduce(loss)
                                count += 1
                                val_running_loss += loss
                            break

                        # print out
                        if WORLD_RANK == 0:
                            train_loss_list.append(
                                running_loss / log_freq / WORLD_SIZE / train_batch_size
                            )
                            val_loss_list.append(
                                val_running_loss
                                / len(test_loader)
                                / count
                                / WORLD_SIZE
                                / test_batch_size
                            )
                            pos = net.module.model.pos_embed.detach().clone()
                            for name, para in net.named_parameters():
                                if "enc.128x128_conv.weight" in name:
                                    conv = para.detach().clone()
                            gpu_memory_used = torch.cuda.memory_allocated() / (
                                1024 * 1024 * 1024
                            )  # Convert to GB
                            toc = time.time()
                            if old_pos is not None and old_pos2 is not None:
                                a = torch.sqrt(torch.sum((pos - old_pos) ** 2))
                                b = torch.sqrt(torch.sum((old_pos - old_pos2) ** 2))
                                corr_pos = (
                                    (
                                        torch.sum(
                                            (pos - old_pos) * (old_pos - old_pos2)
                                        )
                                        / (a * b)
                                    )
                                    .cpu()
                                    .detach()
                                    .numpy()
                                )
                                a = torch.sqrt(torch.sum((conv - old_conv) ** 2))
                                b = torch.sqrt(torch.sum((old_conv - old_conv2) ** 2))
                                corr_conv = (
                                    (
                                        torch.sum(
                                            (conv - old_conv) * (old_conv - old_conv2)
                                        )
                                        / (a * b)
                                    )
                                    .cpu()
                                    .detach()
                                    .numpy()
                                )
                                print(
                                    "  step {:8d} | loss: {:.2e}, val loss: {:.2e}, diff pos: {:.2e}, corr pos: {:.2f}, diff conv: {:.2e}, corr conv: {:.2f}, grad norm: {:.2e}, gpu usage: {:.3f}, time: {:6.1f} sec".format(
                                        step,
                                        train_loss_list[-1],
                                        val_loss_list[-1],
                                        torch.sum(
                                            torch.abs(old_pos - pos) / torch.numel(pos)
                                        )
                                        .cpu()
                                        .detach()
                                        .numpy(),
                                        corr_pos,
                                        torch.sum(
                                            torch.abs(old_conv - conv)
                                            / torch.numel(conv)
                                        )
                                        .cpu()
                                        .detach()
                                        .numpy(),
                                        corr_conv,
                                        grad_norm,
                                        gpu_memory_used,
                                        (toc - tic),
                                    ),
                                    flush=True,
                                )
                            else:
                                print(
                                    "  step {:8d} | loss: {:.2e}, val loss: {:.2e}, grad norm: {:.2e}, gpu usage: {:.3f}, time: {:6.1f} sec".format(
                                        step,
                                        train_loss_list[-1],
                                        val_loss_list[-1],
                                        grad_norm,
                                        gpu_memory_used,
                                        (toc - tic),
                                    ),
                                    flush=True,
                                )
                            if old_pos is not None:
                                old_pos2 = old_pos.detach().clone()
                                old_conv2 = old_conv.detach().clone()
                            old_pos = pos.detach().clone()
                            old_conv = conv.detach().clone()
                            file_name = "cBottle-SR-{}.zip".format(step)
                            save_checkpoint(
                                os.path.join(output_path, file_name),
                                model_config=model_config,
                                network=net,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                step=step,
                                loss=train_loss_list,
                            )
                            running_loss = 0.0

                if step >= num_steps:
                    print("training finished!")
                    return

                # break after a single batch if in testing mode
                scheduler.step()
