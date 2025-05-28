import gc
import torch
from torch import nn, optim
from torch.nn import functional as F


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, temperature=1.5):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * float(temperature))

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1),logits.size(2),logits.size(3)).cuda()

        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, train_loader, valid_loader, lr=0.5, max_iter=20):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()

        nll_criterion = nn.CrossEntropyLoss().cuda()

        total_nll = torch.tensor(0.0, device='cuda')
        total_samples = 0

        with torch.no_grad():
            for step, (input, label, filename, filenameGt) in enumerate(valid_loader):

                # print (step, filename[0].split("leftImg8bit/")[1])

                input = input.cuda()
                label = label.cuda()

                # logits is a pytorch tensor
                logits = self.model(input)

                # Calcola NLL direttamente sul batch corrente
                batch_nll = nll_criterion(logits, label) * input.size(0)

                # Aggiorna il conteggio totale
                # la forma x += y restituisce errori di shape non corrispondenti
                total_nll = total_nll + batch_nll
                total_samples += input.size(0)

                # print(torch.cuda.memory_summary())
                # allocated = torch.cuda.memory_allocated()
                # reserved = torch.cuda.memory_reserved()
                # print(f"Memory Allocated: {allocated / (1024 ** 2):.2f} MB")
                # print(f"Memory Reserved: {reserved / (1024 ** 2):.2f} MB")

                # Libera la memoria non necessaria
                del input, label, logits, batch_nll
                gc.collect()
                torch.cuda.empty_cache()

        # Calculate NLL before temperature scaling
        # before_temperature_nll = total_nll / total_samples
        # print('Before temperature - NLL: %.3f' % (before_temperature_nll))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        # print(f"Temperature before optimization: {self.temperature.item()}")

        call_count = 0

        def eval():

            nonlocal call_count
            call_count += 1
            print(f"Eval called {call_count} times")
            # print(f"Temperature: {self.temperature.item()}")

            optimizer.zero_grad()

            nll_criterion = nn.CrossEntropyLoss().cuda()

            total_loss = 0.0
            total_samples = 0

            for step, (input, label, filename, filenameGt) in enumerate(train_loader):

                # print (step, filename[0].split("leftImg8bit/")[1])

                input = input.cuda()
                label = label.cuda()

                # logits is a pytorch tensor
                with torch.no_grad():
                    logits = self.model(input)

                scaled_logits = self.temperature_scale(logits)

                batch_loss = nll_criterion(scaled_logits, label) * input.size(0)

                total_loss = total_loss + batch_loss
                total_samples += input.size(0)

                # Calcola NLL direttamente sul batch corrente
                #batch_nll = nll_criterion(scaled_logits, label) * input.size(0)

                # Aggiorna il conteggio totale
                # la forma x += y restituisce errori di shape non corrispondenti
                # total_nll = total_nll + batch_nll
                #total_samples += input.size(0)

                # print(torch.cuda.memory_summary())
                # allocated = torch.cuda.memory_allocated()
                # reserved = torch.cuda.memory_reserved()
                # print(f"Memory Allocated: {allocated / (1024 ** 2):.2f} MB")
                # print(f"Memory Reserved: {reserved / (1024 ** 2):.2f} MB")

                # Libera la memoria non necessaria
                del input, label, logits
                gc.collect()
                torch.cuda.empty_cache()
            
            total_loss = total_loss / total_samples

            print(f"Total loss: {total_loss.item()}")

            total_loss.backward()
            
            return total_loss

        optimizer.step(eval)

        # print(f"Temperature after backward: {self.temperature.item()}")
        # Calculate NLL after temperature scaling
        # after_temperature_nll = self.get_nll(valid_loader, temperature_scale=True)
        print('Optimal temperature: %.3f' % self.temperature.item())
        # print('After temperature - NLL: %.3f' % (after_temperature_nll))

        total_nll = torch.tensor(0.0, device='cuda')
        total_samples = 0

        with torch.no_grad():
            for step, (input, label, filename, filenameGt) in enumerate(valid_loader):

                # print (step, filename[0].split("leftImg8bit/")[1])

                input = input.cuda()
                label = label.cuda()

                # logits is a pytorch tensor
                logits = self.model(input)

                logits = self.temperature_scale(logits)

                # Calcola NLL direttamente sul batch corrente
                batch_nll = nll_criterion(logits, label) * input.size(0)

                # Aggiorna il conteggio totale
                # la forma x += y restituisce errori di shape non corrispondenti
                total_nll = total_nll + batch_nll
                total_samples += input.size(0)

                # print(torch.cuda.memory_summary())
                # allocated = torch.cuda.memory_allocated()
                # reserved = torch.cuda.memory_reserved()
                # print(f"Memory Allocated: {allocated / (1024 ** 2):.2f} MB")
                # print(f"Memory Reserved: {reserved / (1024 ** 2):.2f} MB")

                # Libera la memoria non necessaria
                del input, label, logits, batch_nll
                gc.collect()
                torch.cuda.empty_cache()

        # Calculate NLL before temperature scaling
        after_temperature_nll = total_nll / total_samples
        print('After temperature - NLL: %.3f' % (after_temperature_nll))


        return self.temperature.item(), after_temperature_nll.item()



class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece