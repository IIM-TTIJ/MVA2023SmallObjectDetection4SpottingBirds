import os
import json

# https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
ANNOTATION_DIR = 'data/drone/annotation'
CATS = ['hawk', 'crow', 'wild bird']
# categories_info = [{'name': CATS[i], 'id': i + 1} for i in range(len(CATS))]

# merge the three classes into one
# categories_info = [{'name': CATS[i], 'id': 0} for i in range(len(CATS))]
categories_info = [{'name': 'bird', 'id': 0}]
CAT_TO_ID = {CATS[i]: 0 for i in range(len(CATS))}

H, W = 2160, 3840


def main():
    assert os.path.isdir(ANNOTATION_DIR)

    train_json_file = os.path.join(ANNOTATION_DIR, 'train.json')
    val_json_file = os.path.join(ANNOTATION_DIR, 'val.json')

    for input_file in [val_json_file, train_json_file]:
        _, old_ext = os.path.splitext(input_file)
        new_ext = '_coco.json'
        output_file = input_file.replace(old_ext, new_ext, 1)
        mini_output_file = input_file.replace(old_ext, f'_mini{new_ext}', 1)

        ret = {'images': [], 'annotations': [], 'categories': categories_info}

        with open(input_file) as f:
            data = f.read()

        results = json.loads(data)
        num_anns = 0
        for k, rc in enumerate(results):
            image_info = {
                'id': k,  # image id.
                'file_name': rc['path'],
                'width': W,
                'height': H,
            }
            ret['images'].append(image_info)

            # instance information in COCO format
            # https://www.programmersought.com/article/43131872583/
            # dataset['annotations'].append({
            #     'area': width * height,
            #     'bbox': [x1, y1, width, height],
            #     'category_id': int(cls_id),
            #     'id': i,
            #     'image_id': k,
            #     'iscrowd': 0,
            #     # mask, the rectangle is the four vertices clockwise from the top left corner
            #     'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
            # }
            # )
            boxes, labels = rc['bbox'], rc['label']
            for box, label in zip(boxes, labels):
                # The COCO bounding box format is [top left x position, top left y position, width, height].
                # The drone dataset has the same format [x1, y1, w1, h1], e.g., ["2728", "1502", "54", "52"]
                box = [float(x) for x in box]  # str to float
                ann_info = {
                    'id': num_anns,
                    'image_id': k,
                    'bbox': box,
                    'area': box[2] * box[3],
                    'category_id': CAT_TO_ID[label],
                    # iscrowd is always set to 0
                    "iscrowd": 0,
                }
                # append fake segmentation mask
                x1, y1, x2, y2 = box[0], box[1], box[0] + box[2], box[1] + box[3]
                ann_info['segmentation'] = [[x1, y1, x2, y1, x2, y2, x1, y2]]
                ret['annotations'].append(ann_info)
                num_anns += 1

        print('{} {} images {} boxes'.format(
            input_file, len(ret['images']), len(ret['annotations'])))
        print('out_path', output_file)
        json.dump(ret, open(output_file, 'w'))

        num_ann_mini = min(60, len(ret['annotations']))
        image_ids = []
        for i in range(num_ann_mini):
            image_ids.append(ret['annotations'][i]['image_id'])

        ret_mini = {
            'images': [img_info for img_info in ret['images'] if img_info['id'] in image_ids],
            'annotations': ret['annotations'][:num_ann_mini],
            'categories': categories_info,
        }
        print(f'out_path {mini_output_file}')
        json.dump(ret_mini, open(mini_output_file, 'w'))


if __name__ == '__main__':
    main()
