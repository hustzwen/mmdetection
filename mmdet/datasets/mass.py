import numpy as np
import pandas as pd

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class MassDataset(CustomDataset):
    CLASSES = ('mass',)

    def load_annotations(self, ann_file):
        data_infos = []
        df = pd.read_csv(ann_file)
        for index, row in df.iterrows():
            width, height = map(lambda x: int(x), row['img_size'].split('/'))
            bbox_str = row['mass_bbox']
            bboxes = []
            for bbox in bbox_str.split(';'):
                bboxes.append(bbox.split('/'))
            data_infos.append(dict(
                filename=row['png_file'],
                width=width,
                height=height,
                ann=dict(
                    bboxes=np.array(bboxes).astype(np.float32),
                    labels=np.ones(len(bboxes), dtype=np.int64)
                )
            ))

        return data_infos

    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']

