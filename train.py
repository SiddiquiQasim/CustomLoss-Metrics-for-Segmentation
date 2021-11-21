import os
from uNet import uNet
from dataGenerator import dataGenerator
from Loss import focalTverskyLoss
from Matrics import ioU, dice, mAP

dataDir = '/Task02_Heart/Task02_Heart/slices/'
dataDirTrainImgs = os.path.join(dataDir, 'imgs/')
dataDirTrainLabels = os.path.join(dataDir, 'labels/')
dataDirTestImgs = os.path.join(dataDir, 'testImgs/')
dataDirTestLabels = os.path.join(dataDir, 'testLabels/')
outputmodelDir = '/Task02_Heart/Task02_Heart/models/'

NUM_TRAIN = 2161
NUM_TEST = 110
batchSizeTrain = 32
batchSizeTest = 32

EPOCHS_STEP_TRAIN = NUM_TRAIN // batchSizeTrain
EPOCHS_STEP_TEST = NUM_TEST // batchSizeTest

NUM_OF_EPOCHS = 8

generator = dataGenerator(dataDirTrainImgs, dataDirTrainLabels)
train_generator = generator.segmentation((320,320))

generator = dataGenerator(dataDirTestImgs, dataDirTestLabels)
test_generator = generator.segmentation((320,320))

model = uNet(320, 320)
model = model.build(4)
loss =  focalTverskyLoss(alpha=0.25, gamma=2)
iou = ioU()
dice = dice()
mAP = mAP(0.5)
model.compile(optimizer="adam", loss=loss, metrics=['accuracy', iou, dice, mAP])

model.fit_generator(train_generator, validation_data=test_generator,
                    validation_steps=EPOCHS_STEP_TEST, steps_per_epoch=EPOCHS_STEP_TRAIN,
                    epochs=NUM_OF_EPOCHS)

model.save_weights(outputmodelDir)