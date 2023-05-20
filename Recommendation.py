####Data Preparation####
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import matplotlib.image as mpimg
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os # accessing directory structure

DATASET_PATH = "/content/drive/MyDrive/test/data/fashion4/" # specify folder path
print(os.listdir(DATASET_PATH)) # show file names in folder

df = pd.read_csv(DATASET_PATH + "styles.csv", nrows=5000, error_bad_lines=False) # read the csv file from the specified path as a dataframe
df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1) # add a new column based on the id column and name it image
df = df.reset_index(drop=True) # rearrange index
df.head(10) # display the first ten rows of the dataframe

import cv2
def plot_figures(figures, nrows = 1, ncols = 1, figsize = (8, 8)):
    """Plot a dictionary of figures

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows,figsize=figsize)
    for ind,title in enumerate(figures):
        axeslist.ravel()[ind].imshow(cv2.cvtColor(figures[title], cv2.COLOR_BGR2RGB))
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional
    
def img_path(img):
    """Get image path

    Parameters
    ----------
    img : image name
    """

    return DATASET_PATH+"images/"+img

def load_image(img, resized_fac = 3):
    """Get image path

    Parameters
    ----------
    img : image name
    resized_fac ： the size fraction you want to resized
    """

    img = cv2.imread(img_path(img)) # read image
    if type(img) == np.ndarray:
      w, h, _ = img.shape # view image size
      resized = cv2.resize(img, (int(h*resized_fac), int(w*resized_fac)), interpolation = cv2.INTER_LINEAR) # adjust pixel values using linear interpolation
      return resized
    else:
      return img

# search images that can not be read
result = df['image'].apply(load_image)
delt = [i for i in range(len(result)) if type(result[i]) != np.ndarray]
delt

df = df.drop(delt)  # delete images that can not be read
df = df.reset_index(drop=True) # rearrange index
import matplotlib.pyplot as plt
import numpy as np

# generation of a dictionary of (title, images)
figures = {'im'+str(i): load_image(row.image) for i, row in df.sample(6).iterrows()}
# plot of the images in a figure, with 2 rows and 3 columns
plot_figures(figures, 2, 3)

plt.figure(figsize=(7,20))
df.articleType.value_counts().sort_values().plot(kind='barh')

####Use Pre-Trained Model to Recommendation####
import tensorflow as tf
import keras
from keras import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import GlobalMaxPooling2D
tf.__version__

# Input Shape
img_width, img_height, _ = 224, 224, 3 # load_image(df.iloc[0].image).shape

# Pre-Trained Model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape = (img_width, img_height, 3))
base_model.trainable = False

# Add Layer Embedding
model = keras.Sequential([base_model, GlobalMaxPooling2D()])

model.summary()

def get_embedding1(model, img_name):
    # reshape
    img = tf.keras.utils.load_img(img_path(img_name), target_size=(img_width, img_height))
    # img to array
    x = tf.keras.utils.img_to_array(img)
    # expand dim (1, w, h)
    x = np.expand_dims(x, axis=0)
    # pre process input
    x = preprocess_input(x)
    return model.predict(x).reshape(-1)

##Get item Embedding
emb = get_embedding1(model, df.iloc[0].image)
emb.shape

img_array = load_image(df.iloc[0].image)
plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
print(img_array.shape)
print(emb)

df.shape

##Get Embedding for all itens in dataset
# import swifter

# Parallel Apply
df_sample = df#.sample(10)
map_embeddings1 = df_sample['image'].apply(lambda img: get_embedding1(model, img))
df_embs1 = map_embeddings1.apply(pd.Series)

print(df_embs1.shape)
df_embs1.head()

##Compute Similarity Between Items
from sklearn.metrics.pairwise import pairwise_distances

# Calcule Distance Matrix
cosine_sim1 = 1 - pairwise_distances(df_embs1, metric='cosine')
cosine_sim1[:4, :4]

##Recommender Similar Items
indices = pd.Series(range(len(df)), index=df.index)
indices

# function that get movie recommendations based on the cosine similarity score of movie genres
def get_recommender1(idx, df, top_n = 5):
    sim_idx    = indices[idx]
    sim_scores = list(enumerate(cosine_sim1[sim_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    idx_rec    = [i[0] for i in sim_scores]
    idx_sim    = [i[1] for i in sim_scores]
    
    return indices.iloc[idx_rec].index, idx_sim
    
get_recommender1(2993, df, top_n = 5)

idx_ref = 0

# Recommendations
idx_rec, idx_sim = get_recommender1(idx_ref, df, top_n = 6)

# Plot
plt.imshow(cv2.cvtColor(load_image(df.iloc[idx_ref].image), cv2.COLOR_BGR2RGB))

# generation of a dictionary of (title, images)
figures = {'im'+str(i): load_image(row.image) for i, row in df.loc[idx_rec].iterrows()}
# plot of the images in a figure, with 2 rows and 3 columns
plot_figures(figures, 2, 3)



####Use Convolutional Autoencoder to Recommendation####
img = tf.keras.utils.load_img(img_path(df['image'].iloc[0]), target_size=(28, 28))
# img to array
x = tf.keras.utils.img_to_array(img)
# expand dim (1, w, h)
x = np.expand_dims(x, axis=0)
x.shape

def get_x(img_name):
    # reshape
    img = tf.keras.utils.load_img(img_path(img_name), target_size=(28, 28))
    # img to array
    x = tf.keras.utils.img_to_array(img)
    # expand dim (1, w, h)
    x = np.expand_dims(x, axis=0)
    # pre process input
    x = preprocess_input(x)
    x = x.astype('float32')/255.
    return x

df_all = df
df_train = df.sample(frac=0.9, random_state=1234, axis=0)
df_val = df.drop(df_train.index).sample(frac=0.5, random_state=1234, axis=0)
df_test = df.drop(df_train.index).drop(df_val.index)

df_all = df_all['image'].apply(lambda img: get_x(img))
df_train = df_train['image'].apply(lambda img: get_x(img))
df_val = df_val['image'].apply(lambda img: get_x(img))
df_test = df_test['image'].apply(lambda img: get_x(img))

x_all = np.concatenate(df_all.to_list(), axis=0)
x_train = np.concatenate(df_train.to_list(), axis=0)
x_val = np.concatenate(df_val.to_list(), axis=0)
x_test = np.concatenate(df_test.to_list(), axis=0)

import keras
from keras import layers


input_img = keras.Input(shape=(28, 28, 3))

x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)


autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

autoencoder.fit(x_train, x_train,
      epochs = 50,
      batch_size = 128,
      shuffle = True,
      validation_data = (x_val, x_val))

decoded_imgs = autoencoder.predict(x_test)
decoded_imgs.shape

encoder = keras.Model(input_img, encoded)
encoded_img = encoder.predict(x_test)
encoded_img.shape

def get_embedding2(x):
    # flatten
    x = x.flatten()
    # pre process input
    x = preprocess_input(x)
    return x

emb2 = encoder.predict(x_all)
emb2.shape
df_embs2 = [get_embedding2(x) for x in emb2]
df_embs2 = pd.DataFrame(df_embs2)
df_embs2.head()

##Compute Similarity Between Items
from sklearn.metrics.pairwise import pairwise_distances

# Calcule Distance Matrix

cosine_sim2 = 1 - pairwise_distances(df_embs2, metric='cosine')
cosine_sim2[:4, :4]

##Recommender Similar Items
indices = pd.Series(range(len(df)), index=df.index)
indices

def get_recommender2(idx, df, top_n = 5):
    sim_idx    = indices[idx]
    sim_scores = list(enumerate(cosine_sim2[sim_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    idx_rec    = [i[0] for i in sim_scores]
    idx_sim    = [i[1] for i in sim_scores]
    
    return indices.iloc[idx_rec].index, idx_sim
    
get_recommender2(2993, df, top_n = 5)

##Recommender Similar Items
# Idx Item to Recommender
idx_ref = 0

# Recommendations
idx_rec, idx_sim = get_recommender2(idx_ref, df, top_n = 6)

# Plot
plt.imshow(cv2.cvtColor(load_image(df.iloc[idx_ref].image), cv2.COLOR_BGR2RGB))

# generation of a dictionary of (title, images)
figures = {'im'+str(i): load_image(row.image) for i, row in df.loc[idx_rec].iterrows()}
# plot of the images in a figure, with 2 rows and 3 columns
plot_figures(figures, 2, 3)

##Visualization Latent Space of Contents
from sklearn.manifold import TSNE
import time
import seaborn as sns

df.shape

df.head()

df_embs1.shape

df_embs2.shape

time_start = time.time()
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
tsne_results1 = tsne.fit_transform(df_embs1)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df['tsne-2d-one-1'] = tsne_results1[:,0]
df['tsne-2d-two-1'] = tsne_results1[:,1]

time_start = time.time()
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
tsne_results2 = tsne.fit_transform(df_embs2)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df['tsne-2d-one-2'] = tsne_results2[:,0]
df['tsne-2d-two-2'] = tsne_results2[:,1]

plt.figure(figsize=(16,10))
sns.scatterplot(x="tsne-2d-one-1", y="tsne-2d-two-1",
                hue="masterCategory",
                data=df,
                legend="full",
                alpha=0.8)

plt.figure(figsize=(16,10))
sns.scatterplot(x="tsne-2d-one-2", y="tsne-2d-two-2",
                hue="masterCategory",
                data=df,
                legend="full",
                alpha=0.8)

plt.figure(figsize=(16,10))
sns.scatterplot(x="tsne-2d-one-1", y="tsne-2d-two-1",
                hue="subCategory",
                data=df,
                legend="full",
                alpha=0.8)

plt.figure(figsize=(16,10))
sns.scatterplot(x="tsne-2d-one-2", y="tsne-2d-two-2",
                hue="subCategory",
                data=df,
                legend="full",
                alpha=0.8)

