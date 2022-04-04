import foolbox as fb
from config import *
import numpy as np
from utils import *

bounds = [[-1.87320715, 2.04073466], [-1.88481253, 2.14623784], [-1.51516363, 2.36902795]]
epsilon = np.random.random(1) * (cfg['attack_range'][1] - cfg['attack_range'][0]) + cfg['attack_range'][0]


def attack(inputs, model, labels, attack_algo):
    if attack_algo == 'FGSM':
        return FGSM_attack(inputs, model, labels)
    elif attack_algo == 'PGD':
        return PGD_attack(inputs, model, labels)


def PGD_attack(inputs, model, labels):
    model.eval()
    fbmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=None)
    steps = np.random.randint(low=40, high=60)
    pgd_attack = fb.attacks.projected_gradient_descent.LinfProjectedGradientDescentAttack(steps=steps,
                                                                                          random_start=True)

    raw, clipped, is_adv = pgd_attack(model=fbmodel, inputs=inputs, criterion=labels, epsilons=epsilon)

    model.train()
    output = model(clipped[0])
    pgd_loss = critirion(output, labels)

    # 计算攻击成功率
    attack_accuracy = torch.count_nonzero(is_adv)/inputs.size()[0]

    return pgd_loss, attack_accuracy



def FGSM_attack(inputs, model, labels):
    model.eval()
    fbmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=None)
    fgsm_attack = fb.attacks.fast_gradient_method.LinfFastGradientAttack()

    raw, clipped, is_adv = fgsm_attack(model=fbmodel, inputs=inputs, criterion=labels, epsilons=epsilon)

    model.train()
    output = model(clipped[0])
    fgsm_loss = critirion(output, labels)

    # 计算攻击成功率
    attack_accuracy = torch.count_nonzero(is_adv)/inputs.size()[0]

    return fgsm_loss, attack_accuracy
