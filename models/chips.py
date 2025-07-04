class Chips:
    def __init__(self, initial=None):
        """
        Pula żetonów gracza
        """
        if isinstance(initial, int):
            self.stack = self._make_stack(initial)
        elif isinstance(initial, dict):
            self.stack = initial
        else:
            self.stack = {}

    def _make_stack(self, amount):
        denominations = [100, 50, 25, 10, 5, 1]
        result = {}
        for denom in denominations:
            count, amount = divmod(amount, denom)
            if count:
                result[denom] = count
        return result

    def total(self):
        return sum(denom * count for denom, count in self.stack.items())
    
    def remove_exact(self, amount):
        denominations = sorted(self.stack.keys(), reverse=True)
        temp_stack = self.stack.copy()
        result = {}
        remaining = amount

        for denom in denominations:
            max_use = min(temp_stack.get(denom, 0), remaining // denom)
            if max_use > 0:
                result[denom] = max_use
                temp_stack[denom] -= max_use
                remaining -= denom * max_use
            if remaining == 0:
                break

        if remaining == 0:
            self.stack = {d: count for d, count in temp_stack.items() if count > 0}
            return True
        else:
            return False

    def make_change(self, required_amount):
        total = self.total()

        if total < required_amount:
            return False

        denominations = [100, 50, 25, 10, 5, 1]

        for preferred_start in range(len(denominations)):
            result = {}
            remaining = total
            for denom in denominations[preferred_start:]:
                count = remaining // denom
                if count > 0:
                    result[denom] = count
                    remaining -= count * denom
            if remaining > 0:
                continue

            temp_stack = result.copy()
            to_pay = required_amount
            for denom in sorted(temp_stack.keys(), reverse=True):
                use = min(to_pay // denom, temp_stack[denom])
                to_pay -= use * denom
                temp_stack[denom] -= use

            if to_pay == 0:
                self.stack = result
                return True

        return False


    def bet(self, amount):
        
        if self.total() < amount:
            raise ValueError("Za mało żetonów!")

        original_stack = self.stack.copy()

        if self.remove_exact(amount):
            return amount

        self.stack = original_stack.copy()
        if self.make_change(amount):
            if self.remove_exact(amount):
                return amount
            else:
                self.stack = original_stack.copy()

        raise ValueError("Nie da się wypłacić dokładnie tej kwoty z dostępnych żetonów")

    def payout(self, amount):
        new_stack = self._make_stack(amount)
        for denom, count in new_stack.items():
            self.stack[denom] = self.stack.get(denom, 0) + count
