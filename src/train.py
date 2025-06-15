from src.models.convection import ConvectionModel
from src.data.square_domain import SquareDomain2D
from typing import List


def train_conv_model(model: ConvectionModel, optimizer, domain: SquareDomain2D,
                     beta: float, N=1000, epochs=5000) -> List:
    loss_values = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        inter = domain.gen_rand_int(N).to(model.device)
        bnd_left = domain.gen_rand_left_bnd(N).to(model.device)
        bnd_bot, bnd_top = domain.gen_rand_top_bot_bnd(N)
        bnd_bot = bnd_bot.to(model.device)
        bnd_top = bnd_top.to(model.device)

        loss = model.compute_loss(inter, bnd_left, bnd_bot, bnd_top, beta)

        loss.backward()
        optimizer.step()

        loss_values.append(loss.item())

        if epoch % 100 == 99:
            print(f"Loss at epoch {epoch + 1} is: {loss.item()}")
