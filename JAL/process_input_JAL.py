from PIL import Image
import numpy as np

'''
what we want to do in this code is : 

test different ways of how we want our input for our environment to be, example :
    - an image that we can turn to black and white then agents should occupy black pixels, and white pixels should remain empty

ideas : 
    - split the grid into 4/9/16 sub grids so we can reduce the complexity of the model, and dont take into count interactions 
    that wont even happen

''' 

class Input :

    def __init__(self, path, reward = 1, sanction = -1, intermediate = 0.5 , extra_padding = False):
        
        # our input image path
        self.path = path

        # we define what are the rewards and sanctions in our environement
        self.reward = reward
        self.sanction = sanction
        self.inter = intermediate 

        # param if we want to add and extra layer of padding : 
        
        self.extra_padding = extra_padding # this boolean is for adding an extra pooling layer of grey

        self.image = self.pad_black_pixels()

        self.rewards = self.reward_grid()

        self.num_image = np.array(self.rewards)
        self.sub_grids = [self.num_image[x:x+5,y:y+5].tolist() for x in range(0,self.num_image.shape[0],5) for y in range(0,self.num_image.shape[1],5)]         
        
    
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
            If the agent moves to a target black pixel, it gets rewarded. If it moves to a white pixel, the agents gets sanctionned. 
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
