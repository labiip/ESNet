import asyncio
from argparse import ArgumentParser
import numpy as np
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
from datetime import datetime
from mmdet.datasets.coco import CocoDataset
import pycocotools
import pycocotools.mask as mask
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img',help='Image file')
    #parser.add_argument('--img',default='complex/xxm SZ1700333 D41-0 35B.tif' ,help='Image file')
    parser.add_argument('config', help='Config file')
    #parser.add_argument('--config',default='../work_dirs/edge_mask_r101_1/mask_rcnn_r101_fpn_2x_coco.py', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    #parser.add_argument('--checkpoint',default='../work_dirs/edge_mask_r101_1/epoch_50.pth', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    a = datetime.now()
    result = inference_detector(model, args.img)
    # save result boxes and masks
    bboxes = result[0]
    masks = result[1]


    # trans_maks = []
    # for i in masks[0]: # list[list[mask]]
    #     encode_mask = mask.encode(np.asfortranarray(i[0]))
    #     print(encode_mask)
    #     trans_maks.append(encode_mask)


#    np.save('box.npy', bboxes,allow_pickle=True)
#    np.save('mask.npy', masks, allow_pickle=True)

    # print(result[1][0][0].shape)


    b = datetime.now()
    print('speed {} ms'.format((b-a).microseconds))
    #with open('tt.txt','w') as f:
        #for item in result:
          #f.write(str(item)+'/n')
          
    # show the results
    show_result_pyplot(model, args.img, result, score_thr=args.score_thr)


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(model, args.img, result[0], score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        
        main(args)
      
      
