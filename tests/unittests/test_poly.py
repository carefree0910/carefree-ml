from cfml.models import Base
from cfml.misc.data import Data

prices = Data().read("prices.txt")
Base.make("poly", deg=1).fit(prices.x, prices.y).visualize1d(prices.x, prices.y)
