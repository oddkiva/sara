import torch
import torch.torch_version
import torchvision.transforms.v2 as v2
if torch.torch_version.TorchVersion(torch.__version__) < (2, 6, 0):
    from torchvision.io.image import read_image as decode_image
else:
    from torchvision.io.image import decode_image

from oddkiva.brahma.torch.datasets.coco import COCO, Annotation


class COCOObjectDetectionDataset():

    DatasetType = Union[Literal['train'], Literal['val']]

    def __init__(
        self,
        transform: Optional[v2.Transform] = None,
        train_or_val: DatasetType = 'train'
    ):
        self.ds = COCO.make_object_detection_dataset(train_or_val)

    def __len__(self):
        return len(self.ds)

    def vectorize_annotations(
        self,
        annotations: list[Annotation]
    ) -> dict[str, list[Any]]:
       return {
           'boxes': [ann.bbox for ann in annotations],
           'labels': [ann.category_id for ann in annotations]
       }

    # def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
    #     image = decode_image(str(self._image_paths[idx]))
    #     if self._transform is not None:
    #         image_transformed = self._transform(image, annotations)
    #     else:
    #         image_transformed = image, annotations
    #     label = self._image_class_ids[idx]
    #     return image_transformed, label

    # @property
    # def image_class_ids(self) -> list[int]:
    #     return self._image_class_ids

    # def image_class_name(self, idx: int) -> str:
    #     return self._image_labels[idx]
