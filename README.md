# Facebook Marketplace Recommendation Ranking System

Project as part of the AICore Machine Learning Engineering pathway, to build a recommendation ranking system for Facebook marketplace.

## Step 1: Data exploration and cleaning
Data was provided through an EC2 instance containing raw images and tabular data, including information on:
- Product name
- Product description
- Category
- Price
- Location
- Images

The tabular data is imported into pandas and cleaned to import the numerical price data as float values.
The image data is resized to 512 x 512 px and converted to RGB colour channels.

## Step 2: Train a CNN model to classify the category of each product from their images