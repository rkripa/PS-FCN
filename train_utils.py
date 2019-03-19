from models import model_utils
from utils  import time_utils 

def train(args, loader, model, criterion, optimizer, log, epoch, recorder, tensorboard):
    model.train()
    print('---- Start Training Epoch %d: %d batches ----' % (epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync);

    for i, sample in enumerate(loader):
        
        # hack to reduce training time
        if args.iterations != 0 and i >= args.iterations:  break
        
        data  = model_utils.parseData(args, sample, timer, 'train')
        input = model_utils.getInput(args, data)

        out_var = model(input); timer.updateTime('Forward')

        optimizer.zero_grad()
        loss = criterion.forward(out_var, data['tar']); timer.updateTime('Crit');
        criterion.backward(); timer.updateTime('Backward')

        recorder.updateIter('train', loss.keys(), loss.values())

        optimizer.step(); timer.updateTime('Solver')

        iters = i + 1
        if iters % args.train_disp == 0:
            opt = {'split':'train', 'epoch':epoch, 'iters':iters, 'batch':len(loader), 
                    'timer':timer, 'recorder': recorder}
            #rkripa ---
            for tag, value in loss.items():
                tensorboard.scalar_summary(tag, value, iters)
            #rkripa ---

            log.printItersSummary(opt)

    opt = {'split': 'train', 'epoch': epoch, 'recorder': recorder}
    log.printEpochSummary(opt)
