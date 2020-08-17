from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import hydra
from hydra.utils import to_absolute_path
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from src.factory import get_model, get_optimizer, get_scheduler
from src.generator import ImageSequence


@hydra.main(config_path="src/config.yaml")
def main(cfg):
    csv_path = Path(to_absolute_path(__file__)).parent.joinpath("meta", f"{cfg.data.db}.csv")
    df = pd.read_csv(str(csv_path))
    train, val = train_test_split(df, random_state=42, test_size=0.1)
    train_gen = ImageSequence(cfg, train)
    val_gen = ImageSequence(cfg, val)
    model = get_model(cfg)
    opt = get_optimizer(cfg)
    scheduler = get_scheduler(cfg)
    model.compile(optimizer=opt, loss=["categorical_crossentropy", "categorical_crossentropy"],
                  metrics=['accuracy'])
    checkpoint_dir = Path(to_absolute_path(__file__)).parent.joinpath("checkpoint")
    checkpoint_dir.mkdir(exist_ok=True)
    callbacks = [
        LearningRateScheduler(schedule=scheduler),
        ModelCheckpoint(str(checkpoint_dir) + "weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                        monitor="val_loss",
                        verbose=1,
                        save_best_only=True,
                        mode="auto")
    ]

    hist = model.fit(train_gen, epochs=cfg.train.epochs, callbacks=callbacks, validation_data=val_gen)


if __name__ == '__main__':
    main()
