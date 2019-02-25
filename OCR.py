#Install cv2, sclite, mxnet, matplotlib, numpy, gluonnlp, pdf2image
import difflib
import math
import random
import string
import os
from pdf2image import convert_from_path
random.seed(123)
import argparse
import gluonnlp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mxnet as mx
import numpy as np
from skimage import transform as skimage_tf
from utils.iam_dataset import IAMDataset, resize_image, crop_image, crop_handwriting_page
from tqdm import tqdm
from utils.expand_bounding_box import expand_bounding_box
from utils.sclite_helper import Sclite_helper
from utils.word_to_line import sort_bbs_line_by_line, crop_line_images

from paragraph_segmentation_dcnn import make_cnn as ParagraphSegmentationNet, paragraph_segmentation_transform
from word_segmentation import SSD as WordSegmentationNet, predict_bounding_boxes
from handwriting_line_recognition import Network as HandwritingRecognitionNet, handwriting_recognition_transform
from handwriting_line_recognition import decode as decoder_handwriting, alphabet_encoding

ctx = mx.cpu(0)

from utils.CTCDecoder.BeamSearch import ctcBeamSearch
from utils.CTCDecoder.LanguageModel import LanguageModel

from skimage import color
from skimage import io

def get_arg_max(prob):
    '''
    The greedy algorithm convert the output of the handwriting recognition network
    into strings.
    '''
    arg_max = prob.topk(axis=2).asnumpy()
    return decoder_handwriting(arg_max)[0]

def get_beam_search(prob, width=20, k=4):
    possibilities = ctcBeamSearch(prob.softmax()[0].asnumpy(), alphabet_encoding, None, width, k)
    return possibilities[0]

def get_beam_search_with_lm(prob, width=20, k=4):
    lm = LanguageModel('utils/alicewonder.txt', alphabet_encoding)
    possibilities = ctcBeamSearch(prob.softmax()[0].asnumpy(), alphabet_encoding, lm, width, k)
    return possibilities[0]


def recognizer(imge):
    MAX_IMAGE_SIZE_FORM = (1120, 800)
    MAX_IMAGE_SIZE_LINE = (60, 800)
    MAX_IMAGE_SIZE_WORD = (30, 140)


    #test image as arg
    file_path = "sample/"+imge+".png"
    filename, file_ext = os.path.splitext(file_path)
    if file_ext == '.pdf':
        image = convert_from_path(file_path, 500) #PIL image list
        image.save(filename+'.png', 'PNG')

    img = color.rgb2gray(io.imread(file_path))

    image = np.array(img)
    print (image.shape)
    #image = np.array(image)
    #image = np.asarray(image)

    print('Paragraph Segmentation...')

    paragraph_segmentation_net = ParagraphSegmentationNet()
    paragraph_segmentation_net.load_parameters("../models/paragraph_segmentation2.params")

    predicted_bbs = []

    resized_image = paragraph_segmentation_transform(image, MAX_IMAGE_SIZE_FORM)
    bb_predicted = paragraph_segmentation_net(resized_image.as_in_context(ctx))
    bb_predicted = bb_predicted[0].asnumpy()
    bb_predicted = expand_bounding_box(bb_predicted, expand_bb_scale_x=0.05, expand_bb_scale_y=0.07)

    segmented_paragraph_size = (700, 700)

    bb = bb_predicted
    paragraph_segmented_image = crop_handwriting_page(image, bb, image_size=segmented_paragraph_size)

    print('Word Segmentation...')

    word_segmentation_net = WordSegmentationNet(2, ctx=ctx)
    word_segmentation_net.load_parameters("../models/word_segmentation2.params")

    min_c = 0.1
    overlap_thres = 0.1
    topk = 600

    predicted_bb = predict_bounding_boxes(word_segmentation_net, paragraph_segmented_image, min_c, overlap_thres, topk, ctx)

    line_bbs = sort_bbs_line_by_line(predicted_bb, y_overlap=0.4)
    line_images = crop_line_images(paragraph_segmented_image, line_bbs)

    print('Handwriting recognition...')

    handwriting_line_recognition_net = HandwritingRecognitionNet(rnn_hidden_states=512, rnn_layers=2, ctx=ctx, max_seq_len=160)
    handwriting_line_recognition_net.load_parameters("../models/handwriting_line_sl_160_a_512_o_2.params")

    line_image_size = (60, 800)
    form_character_prob = []
    for i, line_image in enumerate(line_images):
        line_image = handwriting_recognition_transform(line_image, line_image_size)
        line_character_prob = handwriting_line_recognition_net(line_image.as_in_context(ctx))
        form_character_prob.append(line_character_prob)



    print('Few more seconds...')

    decodes = ''
    fig, axs = plt.subplots(len(form_character_prob) + 1, figsize=(7, int(1 + 1.2 * len(form_character_prob))))
    for j, line_character_probs in enumerate(form_character_prob):
        decoded_line = get_beam_search_with_lm(line_character_probs)
        decodes = decodes+' '+decoded_line
        line_image = line_images[j]
        axs[j].imshow(line_image.squeeze(), cmap='Greys_r')            
        axs[j].imshow(line_image.squeeze(), cmap='Greys_r')
        axs[j].set_title("{}".format(decoded_line))
        axs[j].axis('off')
    axs[-1].imshow(np.zeros(shape=line_image_size), cmap='Greys_r')
    axs[-1].axis('off')

    plt.show()
    return decodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", action = 'store', dest = 'imge',default='test',help="Input Image")
    args = parser.parse_args()
    decodes = recognizer(args.imge)

if __name__ == '__main__':
    main()


