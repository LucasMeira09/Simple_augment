from PIL import Image  
import os  
import numpy as np

class DataAugmentation:
    
    def __init__(self, directory_path, size_img=255, directory_name="Data_A"):

        # lists all files in the folder
        self.img_directory_list = os.listdir(directory_path)
        
        # folder path
        self.directory_path = directory_path 
        
        # stores the desired image size
        self.size_img = size_img
        
        # name of the folder created when using the object
        self.directory_name = directory_name

        # verification to print error messages only once
        self.already_printed = True

    def save_image(self, output_name, img):
    # saves images
        # name of the folder to be created
        directory_name = self.directory_name
        
        # saves in a new folder, prints message in case of error
        try:
            os.mkdir(directory_name)
            if self.already_printed:
                print(f"Directory '{directory_name}' created successfully.")
                self.already_printed = False
        except FileExistsError:
            if self.already_printed:
                print(f"Directory '{directory_name}' already exists.")
                self.already_printed = False
        except PermissionError:
            if self.already_printed:
                print(f"Permission denied: Unable to create '{directory_name}'.")
                self.already_printed = False
        except Exception as e:
            if self.already_printed:
                print(f'An error occured: {e}')
                self.already_printed = False

        # resizes the image
        img = img.resize((self.size_img, self.size_img))
        # defines the full path to save the new image
        output_path = os.path.join(directory_name, output_name)
        # saves the new rotated image
        img.save(output_path)

    def resize_image(self):
    # resizes images
        # iterates over all files in the folder
        for directory in self.img_directory_list:
            # image path
            img_directory = os.path.join(self.directory_path, directory)
            # opens the image
            img = Image.open(img_directory)
            # resizes the image
            img = img.resize((self.size_img, self.size_img))
            # calls the save function
            self.save_image(directory, img)
        
        
    def Image_rotation(self, random=False, quantity=0, min=10, max=340):
    # applies different rotation levels to images, randomly or as defined by user
        # iterates over each item in the file list
        for directory in self.img_directory_list:

            # builds the full image path
            img_directory = os.path.join(self.directory_path, directory)
            try:
                # opens the image for processing
                img = Image.open(img_directory)

                # defines rotation angles if random
                if random:
                    # random number generator
                    rng = np.random.default_rng()
                    angl_rotate = rng.integers(low= min, high=max, size=quantity)
                else: 
                    # defines list of fixed angles to rotate images
                    angl_rotate = [45, 90, 135, 180, 225, 270, 315]
            
                # for each angle in the list, applies rotation and saves the modified image
                for num,angl in enumerate(angl_rotate):

                    # rotates the image by the specified angle
                    img_rotate = img.rotate(angl, expand=True)

                    # defines the output name for the new image
                    base_name , ext = os.path.splitext(directory)
                    output_name = f"{base_name}_{angl_rotate[num]}{ext}"

                    self.save_image(output_name, img_rotate)

            # captures and prints any errors during the process
            except Exception as e:
                print(f"Error in process {img_directory}: {e}")
                
    def Image_noise(self, sigma):
    # adds noise to the image

        # iterates through the file list
        for directory in self.img_directory_list:
            # creates the path to open the image
            img_directory = os.path.join(self.directory_path, directory)

            # opens the files
            img = Image.open(img_directory)
            # converts to numpy array
            img_array = np.array(img)
            
            # adjusts the noise level
            noise = np.random.normal(0, sigma, img_array.shape).astype(np.int16)
            # adds noise pixel by pixel, limited to not exceed 255
            noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            
            # reconstructs the image
            noisy_image = Image.fromarray(noisy_array)

            # defines the output name for the new image
            base_name , _ = os.path.splitext(directory)
            output_name = f"{base_name}_noisy.jpg"
            # saves the image
            self.save_image(output_name, noisy_image)

    def Image_crop(self, left=0, upper=0, right=0, lower=0, random=False):
    # crops the images randomly if desired or using specific values

        # if condition is true, performs random cropping
        if random == True:
            # iterates through the file list
            for directory in self.img_directory_list:
                # creates the path to open the image
                img_directory = os.path.join(self.directory_path, directory)
                # opens the image
                img = Image.open(img_directory)

                # image dimensions
                width, height = img.size

                # defines random pattern
                rng = np.random.default_rng()
                random = rng.integers(low=1, high=self.size_img, size=4)

                # assigns random values
                crop_w = rng.integers(low=30, high=width)
                crop_h = rng.integers(low=30, high=height)
                
                left = (width - crop_w) // 2
                upper = (height - crop_h) // 2
                right = left + crop_w
                lower = upper + crop_h

                # crops the image
                img = img.crop((left, upper, right, lower))
                # saves
                base_name , _ = os.path.splitext(directory)
                output_name = f"{base_name}_crop.jpg"
                self.save_image(output_name, img)

        elif right > left and lower > upper:
            # iterates through the file list
            for directory in self.img_directory_list:
                # creates the path to open the image
                img_directory = os.path.join(self.directory_path, directory)
                # opens the image
                img = Image.open(img_directory)
                # crops the image
                img = img.crop((left, upper, right, lower))
                # new file name
                base_name , _ = os.path.splitext(directory)
                output_name = f"{base_name}_crop.jpg"
                # saves 
                self.save_image(output_name, img)
        else:
            print(f"Coordinates out of bounds")
                

    def Image_tensorHWC(self):
    # converts images into tensors in HWC format (height, width, channels)
        # tensor list
        tensor_list = []
        
        # iterates through the file list
        for directory in self.img_directory_list:
            img_directory = os.path.join(self.directory_path,directory)
            # opens the files
            img = Image.open(img_directory).convert("RGB")

            # standardizes the image
            img = img.resize((self.size_img, self.size_img))
            # converts file into tensor (H, W, C) and keeps values between 0 and 1
            img_tensor = np.array(img).astype(np.float32) / 255.0
            # appends tensor to list
            tensor_list.append(img_tensor)
        
        # returns a list of tensors for machine training use
        return tensor_list
    
    def Image_tensorCHW(self):    
    # converts images into tensors in CHW format (channels, height, width)
        # tensor list
        tensor_list = []
        
        # iterates through the file list
        for directory in self.img_directory_list:
            img_directory = os.path.join(self.directory_path, directory)
            # opens the files
            img = Image.open(img_directory).convert("RGB")

            # standardizes image to same size
            img = img.resize((self.size_img, self.size_img))

            # converts file into tensor (C, H, W) and keeps values between 0 and 1
            img_tensor = np.array(img).astype(np.float32) / 255.0
            #
            img_tensor3D = np.transpose(img_tensor, (2, 0, 1))
            # appends tensor to list
            tensor_list.append(img_tensor3D)
        
        # returns a list of tensors for machine training use
        return tensor_list
