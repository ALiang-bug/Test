import os
import types
import numpy as np
from utils import get_captions_from_page, save_html
import fasttext

PRETRAINED_MODEL_PATH = os.path.expanduser('~/model_path/lid.176.bin')
model = fasttext.load_model(PRETRAINED_MODEL_PATH)

def patched_predict(self, text, k=1, threshold=0.0, on_unicode_error="strict"):
    def check(entry):
        if entry.find("\n") != -1:
            raise ValueError("predict processes one line at a time (remove '\\n')")
        return entry + "\n"

    if isinstance(text, list):
        text = [check(entry) for entry in text]
        all_labels, all_probs = self.f.multilinePredict(text, k, threshold, on_unicode_error)
        return all_labels, np.asarray(all_probs)
    else:
        text = check(text)
        predictions = self.f.predict(text, k, threshold, on_unicode_error)
        if predictions:
            probs, labels = zip(*predictions)
        else:
            probs, labels = [], ()
        return labels, np.asarray(probs)

model.predict = types.MethodType(patched_predict, model)

def get_captions_and_process(img_url,page_url,page, save_folder_path,file_save_counter,hashing_cutoff):
    caption,title,code,req = get_captions_from_page(img_url, page_url)

    if title is None: title = ''
    page_title = page.page_title if page.page_title else ''
    if len(title) < len(page_title.lstrip().rstrip()):
        title = page_title
    if title:
        lang_pred = model.predict(title.replace("\n"," "))
        if lang_pred[0][0] != '__label__en':
            return{}
    # HTML saving functionality disabled
    # saved_html_flag = save_html(req, os.path.join(save_folder_path,str(file_save_counter)+'.txt'))
    # if saved_html_flag:
    #     html_path = os.path.join(save_folder_path,str(file_save_counter)+'.txt')
    # else:
    #     html_path = ''
    # html_path = ''  # Always set to empty since HTML saving is disabled

    # if caption:
    #     new_entry = {'page_link':page_url,'image_link':img_url,'title': title, 'caption':caption, 'html_path': html_path}
    # else:
    #     caption,title,code,req = get_captions_from_page(img_url,page_url,req,hashing_cutoff)
    #     if caption:
    #         new_entry = {'page_link':page_url,'image_link':img_url, 'caption':caption, 'html_path': html_path, 'matched_image': 1}
    #     else:
    #         new_entry = {'page_link':page_url,'image_link':img_url, 'html_path': html_path}
    if caption:
        new_entry = {'page_link':page_url,'image_link':img_url,'title': title, 'caption':caption}
    else:
        caption,title,code,req = get_captions_from_page(img_url,page_url,req,hashing_cutoff)
        if caption:
            new_entry = {'page_link':page_url,'image_link':img_url, 'caption':caption, 'matched_image': 1}
        else:
            new_entry = {'page_link':page_url,'image_link':img_url}
    if title:
        new_entry['title'] = title
    return new_entry
