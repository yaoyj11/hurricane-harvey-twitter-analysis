from keras import losses
class Config(object):
  def __init__(self):
    self.max_featres=20000
    self.embedding=128
    self.maxlen=80
    batch_size=32
    self.dropout=0.5
    self.reccurrent_dropout=0.5
  
def printConfig(config):
  attrs=vars(config)
  print attrs

def saveConfig(config):
  f=open(config.modelpath+config.modelname+".config","w")
  attrs=vars(config)
  f.write(str(attrs))
  f.close()

def logConfig(config,logfile):
  f=open(logfile,"a")
  attrs=vars(config)
  f.write(str(attrs))
  f.close()

