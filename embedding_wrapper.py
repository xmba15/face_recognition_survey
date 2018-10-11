#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from thirdparty.embedding import IFaceEmbedding


class Embedding(object):
    """
    Embedding Wrapper
    """
    def __init__(self, embedding_class=IFaceEmbedding):
        self.embedding_class = embedding_class()

    def __getattr__(self, attr):
        return self.embedding_class.__getattribute__(attr)
