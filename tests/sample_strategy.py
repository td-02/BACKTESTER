import nanoback as nb


class FlipStrategy(nb.Strategy):
    def on_event(self, event):
        if event.index == 0:
            return [nb.OrderIntent(asset=0, target_position=1)]
        if event.index == 2:
            return [nb.OrderIntent(asset=0, target_position=-1)]
        return ()
