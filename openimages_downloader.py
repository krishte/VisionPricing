import fiftyone as fo
import fiftyone.zoo as foz



export_dir = "datasets/extra_objects"
label_field = "ground_truth"  # for example

# The splits to export
splits = ["train", "validation", "test"]

# All splits must use the same classes list
classes = ["Headphones", "Kettle", "Washing machine", "Printer", "Toaster", "Dishwasher", "Blender"]

dataset = foz.load_zoo_dataset(
              "open-images-v7",
              splits=splits,
              label_types=["detections"],
              classes=[ "Headphones", "Kettle", "Washing machine", "Printer", "Toaster", "Dishwasher", "Blender"],
          ) 

# The dataset or view to export
# We assume the dataset uses sample tags to encode the splits to export

# Export the splits
for split in splits:
    split_view = dataset.match_tags(split)
    split_view.export(
        export_dir=export_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field=label_field,
        split=split,
        classes=classes,
    )


# dataset.export(
#     export_dir="datasets/extra_objects",
#     dataset_type=fo.types.YOLOv5Dataset,
#     label_field="detections",
# )