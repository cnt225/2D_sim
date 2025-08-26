from torch.utils import data
from .Acronym_dataset_rcfm import AcronymDataset4RCFM
from .Motion_dataset_rcfm_ply_used import MotionDataset4RCFM
from .motion_step_dataset import MotionStepDataset


def get_dataloader(data_dict, ddp=False, **kwargs):
    dataset = get_dataset(data_dict)
    if ddp:
        loader = data.DataLoader(
            dataset,
            batch_size=data_dict["batch_size"],
            pin_memory=True,
            shuffle=False,
            sampler=data.distributed.DistributedSampler(dataset)
        )
    else:
        loader = data.DataLoader(
            dataset,
            batch_size=data_dict["batch_size"],
            shuffle=data_dict.get("shuffle", True),
            num_workers=data_dict.get("num_workers", 4)
        )
    return loader

def get_dataset(data_dict):
    name = data_dict["dataset"]
    if name == 'Acronym_rcfm':
        dataset = AcronymDataset4RCFM(**data_dict)
    elif name == 'Motion_dataset_rcfm_ply_used':
        dataset = MotionDataset4RCFM(**data_dict)
    elif name == 'Motion_step_dataset':
        dataset = MotionStepDataset(**data_dict)
    else:
        raise NotImplementedError(f"Dataset {name} is not implemented")
    return dataset