import json
import random
import shutil
from pathlib import Path
from typing import Dict, List
from PIL import Image


def load_json(file_path: str) -> List[Dict]:

    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: List[Dict], file_path: str) -> None:

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def verify_image(image_path: str, base_dir: Path) -> bool:

    full_path = base_dir / image_path
    if not full_path.exists():
        return False
    try:
        with Image.open(full_path) as img:
            img.verify()

        with Image.open(full_path) as img:
            img.load()
        return True
    except Exception:
        return False


def main():

    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)

    metadata_dir = Path("../dataset/DGM4/metadata")
    image_base_dir = Path("../dataset")
    output_dir = Path("DGM4")
    output_dir.mkdir(parents=True, exist_ok=True)

    # load train.json, val.json, test.json

    train_data = load_json(metadata_dir / "train.json")
    val_data = load_json(metadata_dir / "val.json")
    test_data = load_json(metadata_dir / "test.json")

    # merge
    all_data = train_data + val_data + test_data
    print(f"Total sample size: {len(all_data)}")

    fake_cls_mapping = {
        "orig": "original",
        "text_attribute": "textual_veracity_distortion",
        "face_swap": "visual_veracity_distortion",
        "face_attribute": "visual_veracity_distortion",
        "text_swap": "mismatch"
    }

    filtered_data = []
    for sample in all_data:
        fake_cls = sample.get("fake_cls")
        if fake_cls in fake_cls_mapping:
            filtered_data.append({
                "text": sample["text"],
                "image_path": sample["image"],
                "fake_cls": fake_cls_mapping[fake_cls]
            })

    print(f"The number of samples after screening: {len(filtered_data)}")

    # Group by fake_cls
    category_to_samples: Dict[str, List[Dict]] = {}
    for sample in filtered_data:
        category = sample["fake_cls"]
        if category not in category_to_samples:
            category_to_samples[category] = []
        category_to_samples[category].append(sample)

    # Print the number of samples for each category
    for category, samples in category_to_samples.items():
        print(f"  {category}: {len(samples)}")

    # Define the number of samples required for each category
    target_counts = {
        "original": 1800,
        "textual_veracity_distortion": 1800,
        "visual_veracity_distortion": 600,
        "mismatch": 1800
    }

    # Random sampling (with image verification)

    sampled_data = []
    for category, target_count in target_counts.items():
        samples = category_to_samples[category]
        # shuffle
        random.shuffle(samples)

        count = 0
        for sample in samples:
            if count >= target_count:
                break

            if verify_image(sample["image_path"], image_base_dir):
                sampled_data.append(sample)
                count += 1

        if count < target_count:
            print(f"Warning: {category} Insufficient valid images in the category {target_count} , Actual acquisition {count}")
        else:
            print(f"  {category}: sample {count}")

    # shuffle
    random.shuffle(sampled_data)

    print(f"\nThe final dataset sample size: {len(sampled_data)}")

    output_path = output_dir / "processed_dataset.json"
    save_json(sampled_data, str(output_path))
    print(f"The data has been saved to: {output_path}")

    print("\nFinal dataset statistics:")
    final_counts = {}
    for sample in sampled_data:
        cat = sample["fake_cls"]
        final_counts[cat] = final_counts.get(cat, 0) + 1
    for category, count in final_counts.items():
        print(f"  {category}: {count}")

    print("\nStart dividing the training set and the test set...")

    # Group by fake_cls
    category_to_samples_final: Dict[str, List[Dict]] = {}
    for sample in sampled_data:
        category = sample["fake_cls"]
        if category not in category_to_samples_final:
            category_to_samples_final[category] = []
        category_to_samples_final[category].append(sample)

    # Define the number of samples in the training set and the test set
    train_counts = {
        "original": 1500,
        "textual_veracity_distortion": 1500,
        "visual_veracity_distortion": 500,
        "mismatch": 1500
    }

    test_counts = {
        "original": 300,
        "textual_veracity_distortion": 300,
        "visual_veracity_distortion": 100,
        "mismatch": 300
    }

    # Extract the training set and the test set
    train_data = []
    test_data = []

    for category in category_to_samples_final.keys():
        samples = category_to_samples_final[category]
        random.shuffle(samples)

        train_count = train_counts[category]
        test_count = test_counts[category]

        # The first "train_count" items are used as the training set.
        train_data.extend(samples[:train_count])

        # The next "test_count" items will be used as the test set.
        test_data.extend(samples[train_count:train_count + test_count])

        print(f"  {category}: train set: {train_count}, test set: {test_count}")

    # shuffle
    random.shuffle(train_data)
    random.shuffle(test_data)

    # save
    train_output_path = output_dir / "train_final.json"
    test_output_path = output_dir / "test_final.json"

    save_json(train_data, str(train_output_path))
    save_json(test_data, str(test_output_path))

    print(f"\nThe train set has been saved to: {train_output_path}")
    print(f"The test set has been saved to: {test_output_path}")

    print("\ncopy image...")

    dgm4_dir = Path("../dataset/DGM4")
    train_image_dir = dgm4_dir / "train_filtered_image"
    test_image_dir = dgm4_dir / "test_filtered_image"
    train_image_dir.mkdir(parents=True, exist_ok=True)
    test_image_dir.mkdir(parents=True, exist_ok=True)

    def copy_images(data: List[Dict], target_dir: Path, base_dir: Path, dir_prefix: str) -> int:

        copied_count = 0
        for sample in data:
            image_path = sample["image_path"]
            source_path = base_dir / image_path
            if source_path.exists():

                filename = Path(image_path).name

                dest_path = target_dir / filename
                counter = 1
                while dest_path.exists():
                    stem = Path(image_path).stem
                    suffix = Path(image_path).suffix
                    dest_path = target_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
                shutil.copy2(source_path, dest_path)

                sample["image_path"] = f"{dir_prefix}/{dest_path.name}"
                copied_count += 1
            else:
                print(f"warning: the image does not exist - {source_path}")
        return copied_count

    train_copied = copy_images(train_data, train_image_dir, image_base_dir, "train_filtered_image")
    print(f"The train set images have been copied to: {train_image_dir} ({train_copied} )")

    # 复制测试集图片并更新路径
    test_copied = copy_images(test_data, test_image_dir, image_base_dir, "test_filtered_image")
    print(f"The test set images have been copied to: {test_image_dir} ({test_copied} )")

    save_json(train_data, str(train_output_path))
    save_json(test_data, str(test_output_path))
    print(f"\nThe image_path has been updated and the JSON file has been re-saved.")

    print("\nTraining set statistics:")
    train_final_counts = {}
    for sample in train_data:
        cat = sample["fake_cls"]
        train_final_counts[cat] = train_final_counts.get(cat, 0) + 1
    for category, count in train_final_counts.items():
        print(f"  {category}: {count}")

    print("\nTest set statistics:")
    test_final_counts = {}
    for sample in test_data:
        cat = sample["fake_cls"]
        test_final_counts[cat] = test_final_counts.get(cat, 0) + 1
    for category, count in test_final_counts.items():
        print(f"  {category}: {count}")


if __name__ == "__main__":
    main()
