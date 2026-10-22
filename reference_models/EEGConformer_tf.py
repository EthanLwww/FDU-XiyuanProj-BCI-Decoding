import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ELU, AveragePooling2D, Dropout, Dense, LayerNormalization, MultiHeadAttention, Flatten, Reshape, Add, GlobalAveragePooling1D

def EEGConformer(nb_classes, Chans=22, Samples=1000, emb_size=40, depth=6, num_heads=10, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
    """
    TensorFlow Implementation of EEG Conformer
    Based on the PyTorch implementation in the EEG-Conformer repo.
    """
    
    # Input shape: (Chans, Samples, 1)
    input_layer = Input(shape=(Chans, Samples, 1))
    
    # --- Patch Embedding (ShallowNet) ---
    # Reduced kernel size to preserve more time points for short windows (e.g., 1s @ 100Hz)
    x = Conv2D(40, (1, 15), padding='valid', name='temporal_conv', kernel_constraint=tf.keras.constraints.max_norm(1.0))(input_layer)
    x = Conv2D(40, (Chans, 1), padding='valid', name='spatial_conv', kernel_constraint=tf.keras.constraints.max_norm(1.0))(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    
    # Reduced pooling size and strides to increase the sequence length (time steps) for Transformer
    # For 100 samples, after Conv (86 left), pool (1, 8) with stride 8 gives ~10 time steps.
    x = AveragePooling2D(pool_size=(1, 8), strides=(1, 8))(x)
    x = Dropout(drop_p)(x)
    
    # --- Projection ---
    # PyTorch: Conv2d(40, emb_size, (1, 1)) -> Rearrange('b e h w -> b (h w) e')
    x = Conv2D(emb_size, (1, 1), padding='valid', name='projection_conv', kernel_constraint=tf.keras.constraints.max_norm(1.0))(x)
    # Post-pooling shape is (1, Samples_new, emb_size). We want (Samples_new, emb_size)
    # x shape here is (Batch, 1, W_new, emb_size)
    x = Reshape((-1, emb_size))(x) # (Batch, W_new, emb_size)
    
    # --- Transformer Encoder ---
    for i in range(depth):
        # Multi-Head Attention Block
        res = x
        x = LayerNormalization()(x)
        x = MultiHeadAttention(num_heads=num_heads, key_dim=emb_size // num_heads)(x, x)
        x = Dropout(drop_p)(x)
        x = Add()([res, x])
        
        # Feed Forward Block
        res = x
        x = LayerNormalization()(x)
        x = Dense(emb_size * forward_expansion, activation='gelu', kernel_constraint=tf.keras.constraints.max_norm(0.5), kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = Dropout(forward_drop_p)(x)
        x = Dense(emb_size, kernel_constraint=tf.keras.constraints.max_norm(0.5), kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = Dropout(drop_p)(x)
        x = Add()([res, x])
        
    # --- Classification Head ---
    # The original implementation uses Flatten then Dense.
    # Note: The number of features after flattening depends on input Samples.
    x = Flatten()(x)
    
    # Based on the PyTorch ClassificationHead's self.fc:
    # self.fc = nn.Sequential(
    #     nn.Linear(2440, 256),
    #     nn.ELU(),
    #     nn.Dropout(0.5),
    #     nn.Linear(256, 32),
    #     nn.ELU(),
    #     nn.Dropout(0.3),
    #     nn.Linear(32, 4)
    # )
    
    x = Dense(256, kernel_constraint=tf.keras.constraints.max_norm(0.5), kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(32, kernel_constraint=tf.keras.constraints.max_norm(0.5), kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = ELU()(x)
    x = Dropout(0.3)(x)
    
    output_layer = Dense(nb_classes, activation='softmax', kernel_constraint=tf.keras.constraints.max_norm(0.5), kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    
    return Model(inputs=input_layer, outputs=output_layer)
