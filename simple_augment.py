from PIL import Image  
import os  
import numpy as np

class DataAugmentation:
    
    def __init__(self, directory_path, size_img=255, directory_name="Data_A"):

        # lista todos os arquivos da pasta
        self.img_directory_list = os.listdir(directory_path)
        
        # camino da pasta
        self.directory_path = directory_path 
        
        # armazena o tamanho da imagem desejado
        self.size_img = size_img
        
        # nome da pasta criada ao fazer uso dos objetos
        self.directory_name = directory_name

        # verificacao para mandar as mensagens de erros uma so vez
        self.already_printed = True

    def save_image(self, output_name, img):
        # nome da pasta que vai ser criada
        directory_name = self.directory_name

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

        #redimensiona a image
        img = img.resize((self.size_img, self.size_img))
        # Define o caminho completo para salvar a nova imagem
        output_path = os.path.join(directory_name, output_name)
        # Salva a nova imagem rotacionada
        img.save(output_path)

    def resize_image(self):
        #precore todos os arquivos da pasta
        for directory in self.img_directory_list:
            #caminho das imagens
            img_directory = os.path.join(self.directory_path, directory)
            # abre a imagem
            img = Image.open(img_directory)
            # redimensiona a imagem
            img = img.resize((self.size_img, self.size_img))
            # chama a funcao para salvar
            self.save_image(directory, img)
        
        
    def Image_rotation(self, random=False, quantity=0, min=10, max=340):
        
        # Itera sobre cada item da lista de arquivos
        for directory in self.img_directory_list:

            # Monta o caminho completo da imagem
            img_directory = os.path.join(self.directory_path, directory)
            try:
                # Abre a imagem para processamento
                img = Image.open(img_directory)

                # define os angolos de rotacao caso aleatorio
                if random:
                    # padrao de numero aleatorio 
                    rng = np.random.default_rng()
                    angl_rotate = rng.integers(low= min, high=max, size=quantity)
                else: 
                    # Define a lista de ângulos de rotação para aplicar às imagens
                    angl_rotate = [45, 90, 135, 180, 225, 270, 315]
            
                # Para cada ângulo na lista, aplica rotação e salva imagem modificada
                for num,angl in enumerate(angl_rotate):

                    # Rotaciona a imagem no ângulo especificado
                    img_rotate = img.rotate(angl, expand=True)

                    # Define o nome de saída da nova imagem
                    base_name , ext = os.path.splitext(directory)
                    output_name = f"{base_name}_{angl_rotate[num]}{ext}"

                    self.save_image(output_name, img_rotate)

            # Captura e imprime qualquer erro ocorrido durante o processo
            except Exception as e:
                print(f"Error in process {img_directory}: {e}")
                
    def Image_noise(self, sigma):

        # percore a lista de arquivos
        for directory in self.img_directory_list:
            # cria o caminho para abrir a imagem
            img_directory = os.path.join(self.directory_path, directory)

            # abre os arquivos
            img = Image.open(img_directory)
            # transforma em array numpy
            img_array = np.array(img)
            
            # ajusta o nivel do roido
            noise = np.random.normal(0, sigma, img_array.shape).astype(np.int16)
            # adiciona o roido pixel por pixel com limite para nao passar de 255
            noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            
            # reconstroi a imagem
            noisy_image = Image.fromarray(noisy_array)

            # Define o nome de saída da nova imagem
            base_name , _ = os.path.splitext(directory)
            output_name = f"{base_name}_noisy.jpg"
            
            self.save_image(output_name, noisy_image)

    def Image_crop(self, left=0, upper=0, right=0, lower=0, random=False):

        # se a condicao for verdadeira gera os cortes aleatoriamente
        if random == True:
            # percore a lista de arquivos
            for directory in self.img_directory_list:
                # cria o caminho para abrir a imagem
                img_directory = os.path.join(self.directory_path, directory)
                # abre a imagem
                img = Image.open(img_directory)

                # tamanho da imagem
                width, height = img.size

                # define por padra a aleatoridade
                rng = np.random.default_rng()
                random = rng.integers(low=1, high=self.size_img, size=4)

                # atribui os numeros aleatorios
                crop_w = rng.integers(low=30, high=width)
                crop_h = rng.integers(low=30, high=height)
                
                left = (width - crop_w) // 2
                upper = (height - crop_h) // 2
                right = left + crop_w
                lower = upper + crop_h

                # recorta a imagem
                img = img.crop((left, upper, right, lower))
                #salva
                base_name , _ = os.path.splitext(directory)
                output_name = f"{base_name}_crop.jpg"
                self.save_image(output_name, img)

        elif right > left and lower > upper:
            #percore a lista de arquivo
            for directory in self.img_directory_list:
                # cria o caminho para abrir a imagem
                img_directory = os.path.join(self.directory_path, directory)
                # abre a imagem
                img = Image.open(img_directory)
                # recorta a imagem
                img = img.crop((left, upper, right, lower))
                # novo nome do arquivo
                base_name , _ = os.path.splitext(directory)
                output_name = f"{base_name}_crop.jpg"
                #salva 
                self.save_image(output_name, img)
        else:
            print(f"Cordenadas fora do limite")
                

    def Image_tensorHWC(self):
        tensor_list = []
        
        # percorre a lista de arquivos
        for directory in self.img_directory_list:
            img_directory = os.path.join(self.directory_path,directory)
            # abre os arquivos
            img = Image.open(img_directory).convert("RGB")

            # padroniza as imagem
            img = img.resize((self.size_img, self.size_img))
            # transforma o arquivo en tensor (H, W, C) (255, 255, 3) e mantem os valores entre 0 e 1
            img_tensor = np.array(img).astype(np.float32) / 255.0
            # adiciona os tensor em uma lista
            tensor_list.append(img_tensor)
        
        # retorna uma lista de tensores para uso de treinamento de maquina
        return tensor_list
    
    def Image_tensorCHW(self):    
        # lista de tensor
        tensor_list = []
        
        # percorre a lista de arquivos
        for directory in self.img_directory_list:
            img_directory = os.path.join(self.directory_path, directory)
            # abre os arquivos
            img = Image.open(img_directory).convert("RGB")

            # padronisa as imagem em um mesmo tamanho
            img = img.resize((self.size_img, self.size_img))

            # transforma o arquivo en tensor (C, H, W) (3, 255, 255) e mantem os valores entre 0 e 1
            img_tensor = np.array(img).astype(np.float32) / 255.0
            #
            img_tensor3D = np.transpose(img_tensor, (2, 0, 1))
            # adiciona os tensor em uma lista
            tensor_list.append(img_tensor3D)
        
        # retorna uma lista de tensores para uso de treinamento de maquina
        return tensor_list
