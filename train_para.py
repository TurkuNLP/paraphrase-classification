import para_model
import para_data
import pytorch_lightning as pl

if __name__=="__main__":
    data=para_data.PARADataModule(".",30)
    model=para_model.PARAModel()
    trainer=pl.Trainer(gpus=1,val_check_interval=0.25,num_sanity_val_steps=5)
    trainer.fit(model,datamodule=data)


