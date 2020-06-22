from __future__ import print_function
'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import os
import sys
import torch
import random
import argparse
import numpy as np
from GPT2.model import (GPT2LMHeadModel)
from GPT2.utils import load_weight
from GPT2.config import GPT2Config
from GPT2.sample import sample_sequence
from GPT2.encoder import get_encoder
import json
from flask import Flask
from oauthlib.common import generate_client_id as oauthlib_generate_client_id
from oauthlib.common import UNICODE_ASCII_CHARACTER_SET
import arrow
import flask
import pickle
import os.path
import arrow
import httplib2
import io
import shutil
import requests
import json
from flask import Flask, request, session, g, redirect, url_for, abort, \
     render_template, flash, Markup, send_from_directory, escape
# EB looks for an 'application' callable by default.
application = Flask(__name__)

class BaseHashGenerator(object):
    """
    All generators should extend this class overriding `.hash()` method.
    """
    def hash(self):
        raise NotImplementedError()



class DiscountGenerator(BaseHashGenerator):
    def hash(self):
        """
        Generate a client_id for Basic Authentication scheme without colon char
        as in http://tools.ietf.org/html/rfc2617#section-2
        """
        return oauthlib_generate_client_id(length=10, chars=UNICODE_ASCII_CHARACTER_SET)



def generate_discount_code():
    """
    Generate a suitable client id
    """
    client_id_generator = DiscountGenerator()
    return client_id_generator.hash()


class IDGenerator(BaseHashGenerator):
    def __init__(self, lng):
        self.lng = lng

    def hash(self):
        """
        Generate a client_id for Basic Authentication scheme without colon char
        as in http://tools.ietf.org/html/rfc2617#section-2
        """
        return oauthlib_generate_client_id(length=self.lng, chars=UNICODE_ASCII_CHARACTER_SET)


def id_generator(str, lng):
    client_id_generator = IDGenerator(lng)
    return ( str + '_' + client_id_generator.hash())


def increment_votes(current_value):
    return current_value + 1 if current_value else 1


def text_generator(state_dict, data):
    parser = argparse.ArgumentParser()
    #parser.add_argument("--text", type=str, required=True)
    #data['text']
    if 'quiet' not in data:
        data['quiet'] = True

    #parser.add_argument("--quiet", type=bool, default=False)
    #data['quiet']
    #parser.add_argument("--nsamples", type=int, default=1)
    #data['nsamples']
    #parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
    data['unconditional'] = False
    #parser.add_argument("--batch_size", type=int, default=-1)
    data['batch_size'] = 1
    #parser.add_argument("--length", type=int, default=-1)
    data['temperature'] = 0.7
    #parser.add_argument("--temperature", type=float, default=0.7)
    data['top_k'] = 40


    if data['batch_size'] == -1:
        data['batch_size'] = 1
    assert data['nsamples'] % data['batch_size'] == 0

    seed = random.randint(0, 2147483647)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    enc = get_encoder()
    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    model = load_weight(model, state_dict)
    model.to(device)
    model.eval()


    context_tokens = enc.encode(data['text'])

    generated = 0
    response_list = []
    for _ in range(data['nsamples'] // data['batch_size']):
        out = sample_sequence(
            model=model, length=data['length'],
            context=context_tokens  if not  data['unconditional'] else None,
            start_token=enc.encoder['<|endoftext|>'] if data['unconditional'] else None,
            batch_size=data['batch_size'],
            temperature=data['temperature'], top_k=data['top_k'], device=device
        )
        out = out[:, len(context_tokens):].tolist()
        for i in range(data['batch_size']):
            generated += 1
            text = enc.decode(out[i])
            response_list.append(text)
    return response_list

# if __name__ == '__main__':
#     if os.path.exists('gpt2-pytorch_model.bin'):
#         state_dict = torch.load('gpt2-pytorch_model.bin', map_location='cpu' if not torch.cuda.is_available() else None)
#         text_generator(state_dict)
#     else:
#         print('Please download gpt2-pytorch_model.bin')
#         sys.exit()



@application.route('/', methods=['POST', 'OPTIONS', 'GET'])
def predict_text():
    if request.path == '/' or request.path == '':
        if request.method == 'GET':
            response = flask.jsonify({})
            response.headers.set('Access-Control-Allow-Origin', '*')
            response.headers.set('Access-Control-Allow-Credentials', 'true')
            response.headers.set('Access-Control-Allow-Headers', 'Content-Type')
            response.headers.set('Access-Control-Max-Age', '3600')
            response.headers.set('Access-Control-Allow-Methods', 'OPTIONS, GET, POST')
            return response
        if request.method == 'OPTIONS':
            response = flask.jsonify({})
            response.headers.set('Access-Control-Allow-Origin', '*')
            response.headers.set('Access-Control-Allow-Credentials', 'true')
            response.headers.set('Access-Control-Allow-Headers', 'Content-Type')
            response.headers.set('Access-Control-Max-Age', '3600')
            response.headers.set('Access-Control-Allow-Methods', 'OPTIONS, GET, POST')
            return response
        elif request.method == 'POST':

            data = json.loads(request.data)

            state_dict = torch.load('gpt2-pytorch_model.bin',map_location='cpu' if not torch.cuda.is_available() else None)


            generated_text = text_generator(state_dict, data)





            response = flask.jsonify({"suggestion": generated_text})
            response.headers.set('Access-Control-Allow-Origin', '*')
            response.headers.set('Access-Control-Allow-Credentials', 'true')
            response.headers.set('Access-Control-Allow-Headers', 'Content-Type')
            response.headers.set('Access-Control-Max-Age', '3600')
            response.headers.set('Access-Control-Allow-Methods', 'OPTIONS, GET, POST')
            return response

        else:
            return 'Method not supported', 405

    return 'Hello World!'



# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    #application.debug = True
    application.debug = True
    port = int(os.environ.get("PORT", 80))
    application.run(host='0.0.0.0', port=port, debug=True)
