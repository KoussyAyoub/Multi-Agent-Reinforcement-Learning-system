import cv2
from PIL import Image
import numpy as np

'''
what we want to do in this code is : 

    - an image that we can turn to black and white then agents should occupy black pixels, and white pixels should remain empty
    - add grey padding for intermediate rewards
''' 

class Input :

    def __init__(self, path, reward = 100, sanction = -10, intermediate = 10 , extra_padding = False):
        
        # our input image path
        self.path = path

        # Let's read our img then turn it to black and white
        tmp_img = cv2.imread(path, cv2.IMREAD_COLOR) #

        bl_wh_img = cv2.threshold(tmp_img, 125, 255, cv2.THRESH_BINARY)[1]
        
        # now we retrieve the size the our grid for our environment
        self.grid_size = tmp_img.shape[0]
        img = []
        for i in range(self.grid_size) :
            l = [max(item) for item in bl_wh_img[i]]
            img.append(l)
        self.img = img

        # we define what are the rewards and sanctions in our environement
        self.reward = reward
        self.sanction = sanction
        self.inter = intermediate 

        # param if we want to add and extra layer of padding : 
        self.extra_padding = extra_padding

        self.image = self.pad_black_pixels()


    
    def pad_black_pixels(self, grey_value = 180, padding_size = 1):
        # Load the image
        image = Image.open(self.path).convert("L")  # Convert to grayscale
        width, height = image.size

        # Create a copy of the image
        padded_image = image.copy()

        # Get the pixel data
        pixels = padded_image.load()

        # Iterate over each pixel in the image
        for y in range(height):
            for x in range(width):
                # Check if the pixel is black
                if pixels[x, y] == 0:
                    # Apply padding to the neighboring pixels
                    for i in range(-padding_size, padding_size + 1):
                        for j in range(-padding_size, padding_size + 1):
                            # Check if the neighboring pixel is within bounds
                            if 0 <= x + i < width and 0 <= y + j < height:
                                # Add padding to the neighboring pixel if it's not already black
                                if pixels[x + i, y + j] != 0:
                                    pixels[x + i, y + j] = grey_value

        if self.extra_padding :
            # Iterate over each pixel in the image
            for y in range(height):
                for x in range(width):
                    # Check if the pixel is black
                    if pixels[x, y] == grey_value:
                        # Apply padding to the neighboring pixels
                        for i in range(-padding_size, padding_size + 1):
                            for j in range(-padding_size, padding_size + 1):
                                # Check if the neighboring pixel is within bounds
                                if 0 <= x + i < width and 0 <= y + j < height:
                                    # Add padding to the neighboring pixel if it's not already black
                                    if pixels[x + i, y + j] != 0 and pixels[x + i, y + j] != grey_value:
                                        pixels[x + i, y + j] = grey_value + 20
        
        return padded_image


    def reward_grid(self):
        '''
            If the agent is in a target black pixel, it gets rewarded. If its in a white pixel, the agents gets sanctionned. 
        '''
        
        reward_list = []
        width, height = self.image.size 
        
        pixels = self.image.load()

        reward_list = []
        for x in range(height):
            row = []
            for y in range(width): 
                if pixels[x, y] == 0 :     
                    row.append([self.reward, True])
                elif pixels[x, y] == 255 :     
                    row.append([self.sanction, False])
                elif pixels[x, y] == 180 :
                    row.append([self.inter, False])
                else :
                    row.append([self.inter / 2, False])

            reward_list.append(row)
            
        return reward_list       
