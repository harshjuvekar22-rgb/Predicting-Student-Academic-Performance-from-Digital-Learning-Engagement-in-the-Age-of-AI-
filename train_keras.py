"""OPTIONAL: the same deep regression network in Keras/TensorFlow.
NOTE: mirrors train.py but was NOT executed in the authoring sandbox (TensorFlow unavailable there).
Run locally (pip install -r requirements-keras.txt) and check results first. The app does not need it.
Run:  python train_keras.py
"""
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from edu_model import MODEL_DIR, SEED, TARGET, infer_columns, load_data, make_preprocessor

tf.keras.utils.set_random_seed(SEED)
df = load_data().dropna(subset=[TARGET])
numeric, categorical = infer_columns(df)
X_tr, X_te, y_tr, y_te = train_test_split(df[numeric + categorical], df[TARGET], test_size=0.2, random_state=SEED)

prep = make_preprocessor(numeric, categorical)
Xtr, Xte = prep.fit_transform(X_tr), prep.transform(X_te)
Xtr = Xtr.toarray() if hasattr(Xtr, "toarray") else Xtr
Xte = Xte.toarray() if hasattr(Xte, "toarray") else Xte
ysc = StandardScaler()
ytr = ysc.fit_transform(y_tr.to_numpy().reshape(-1, 1)).ravel()

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(Xtr.shape[1],)),
    tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1),
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
model.fit(Xtr, ytr, validation_split=0.1, epochs=200, batch_size=64, verbose=2,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)])
p = ysc.inverse_transform(model.predict(Xte).reshape(-1, 1)).ravel()
print(f"Keras DNN  MAE={mean_absolute_error(y_te, p):.3f}  RMSE={np.sqrt(mean_squared_error(y_te, p)):.3f}  "
      f"R2={r2_score(y_te, p):.4f}")
MODEL_DIR.mkdir(exist_ok=True)
model.save(MODEL_DIR / "dnn_keras.keras")
