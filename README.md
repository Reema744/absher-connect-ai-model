# Absher Connect – AI Context Model

This repository contains a small, lightweight AI model written in Python.

The goal of the model is to decide **when to show a proactive contextual card** to the user inside Absher, based on:

- How many days are left until a document expires.
- Whether the user is near the relevant government office.
- Whether the user has recently renewed the same document.
- The importance of the document type (passport, national ID, driving license, etc.).

The model is implemented as a logistic-style contextual model.  
It exposes:

- `predict_probability(context)` → returns a probability from 0.0 to 1.0  
- `should_show_card(context, threshold=0.6)` → returns True/False

Later, it can be trained on real data using the `fit()` method.

This module is designed to be plugged into any backend (Replit, microservice, or Absher internal APIs).
