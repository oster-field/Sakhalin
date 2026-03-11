# Sakhalin Wave Data Pipeline

## Overview
End-to-end data processing methodology for extracting statistical 
features from large-scale pressure sensor recordings. Developed as 
part of research into rogue wave probability — published in 
Atmospheric and Oceanic Physics, Springer (2025).

## Problem
Raw sensor data contains tidal interference, noise, and irregular 
sampling. Standard approaches fail on broad-spectrum recordings. 
This pipeline implements a domain-driven methodology to clean, 
segment, and extract representative statistics from 2.1B+ data points.

## Methodology
- FFT-based tidal filtering
- 20-minute segmentation with zero-crossing wave detection  
- Extraction of 20+ statistical features per segment
- Probability distribution analysis for extreme events

## Publication
Results published in: Atmospheric and Oceanic Physics · Springer · 
Oct 2025 · DOI: 10.1134/S0001433825700884

## Requirements
See requirements.txt
