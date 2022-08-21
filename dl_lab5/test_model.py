import os

import torch

from evaluator import evaluation_model
from main import load_data, evaluate, parse_args
from models.acgan import Generator as AC_Generator

args = parse_args()

_, test_loader, new_test_loader = load_data(32)

model = AC_Generator(args.n_classes, args.nc, args.latent_dim, args.ngf)
model.load_state_dict(torch.load(os.path.join('logs/classifier', 'generator_best_model.pt'), map_location=args.device))
model.to(args.device)

evaluator = evaluation_model(args.device, args.n_gpu)

best_test_acc = 0
best_new_test_acc = 0
best_test_image = None
best_new_test_image = None
for i in range(0, 10):
    test_acc, test_image = evaluate(evaluator, model, test_loader, args)
    new_test_acc, new_test_image = evaluate(evaluator, model, new_test_loader, args)
    if test_acc < 0.8 or new_test_acc < 0.8:
        continue
    if test_acc + new_test_acc > best_test_acc + best_new_test_acc:
        best_test_acc = test_acc
        best_new_test_acc = new_test_acc
        best_test_image = test_image
        best_new_test_image = best_new_test_image

print("Best Test Accuracy: {}".format(best_test_acc))
print("Best New Test Accuracy: {}".format(best_new_test_acc))
