# Absher Connect - Simple AI model for contextual suggestions
# يحتوي على مودل صغير + إمكانية تدريب بسيطة باستخدام gradient descent.

import math


class AbsherConnectAIModel:
    """
    Mini logistic model:
    score = sigmoid(w1*x1 + w2*x2 + w3*x3 + w4*x4 + bias)
    """

    def __init__(self):
        # Initial weights (can be tuned or trained)
        self.weight_urgency = 2.5
        self.weight_near_office = 2.0
        self.weight_recent_renewal = -1.5
        self.weight_doc_priority = 1.0
        self.bias = -1.5

    # --------- helpers ---------

    def _sigmoid(self, z: float) -> float:
        return 1.0 / (1.0 + math.exp(-z))

    def _calculate_urgency(self, days_left: int | None) -> float:
        """Normalize days until expiry → value between 0 and 1."""
        if days_left is None:
            return 0.0
        clamped = min(max(days_left, 0), 60)
        return 1.0 - (clamped / 60.0)

    def _document_priority(self, doc_type: str | None) -> float:
        """Assign priority based on document type."""
        if not doc_type:
            return 0.5
        d = doc_type.lower()
        if d in ("passport", "national_id", "id"):
            return 1.0
        if d in ("driving_license", "license"):
            return 0.8
        return 0.5

    def _build_features(self, context: dict) -> list:
        """Convert raw context → feature vector."""
        urgency = self._calculate_urgency(context.get("days_until_expiry"))
        near_office = float(context.get("near_relevant_office", 0))
        recent = float(context.get("recent_renewal", 0))
        priority = self._document_priority(context.get("document_type"))
        return [urgency, near_office, recent, priority]

    # --------- prediction ---------

    def predict_probability(self, context: dict) -> float:
        """Return probability (0–1) to show the notification card."""
        u, near, recent, prio = self._build_features(context)

        z = (
            u * self.weight_urgency +
            near * self.weight_near_office +
            recent * self.weight_recent_renewal +
            prio * self.weight_doc_priority +
            self.bias
        )

        return self._sigmoid(z)

    def should_show_card(self, context: dict, threshold: float = 0.6) -> bool:
        return self.predict_probability(context) >= threshold

    # --------- training (fit) ---------

    def fit(self, X: list, y: list, lr: float = 0.1, epochs: int = 200):
        """
        Very small gradient descent training.
        X: list of feature vectors (already transformed)
        y: list of labels 0/1
        """

        for _ in range(epochs):
            grad_w_u = 0.0
            grad_w_near = 0.0
            grad_w_recent = 0.0
            grad_w_prio = 0.0
            grad_b = 0.0

            for features, label in zip(X, y):
                u, near, recent, prio = features

                # forward pass
                z = (
                    u * self.weight_urgency +
                    near * self.weight_near_office +
                    recent * self.weight_recent_renewal +
                    prio * self.weight_doc_priority +
                    self.bias
                )
                pred = self._sigmoid(z)

                # error term
                error = pred - label  # derivative of binary cross-entropy

                # accumulate gradients
                grad_w_u += error * u
                grad_w_near += error * near
                grad_w_recent += error * recent
                grad_w_prio += error * prio
                grad_b += error

            # update weights
            m = len(X)
            self.weight_urgency -= lr * (grad_w_u / m)
            self.weight_near_office -= lr * (grad_w_near / m)
            self.weight_recent_renewal -= lr * (grad_w_recent / m)
            self.weight_doc_priority -= lr * (grad_w_prio / m)
            self.bias -= lr * (grad_b / m)


# Quick test
if __name__ == "__main__":
    model = AbsherConnectAIModel()

    # Example simulated training data (dummy)
    X = [
        [0.9, 1, 0, 1],   # urgent, near office → should show
        [0.1, 0, 0, 1],   # not urgent → don't show
        [0.8, 1, 1, 1],   # urgent but recently renewed
    ]
    y = [1, 0, 0]

    model.fit(X, y, lr=0.1, epochs=100)

    test_ctx = {
        "days_until_expiry": 10,
        "near_relevant_office": 1,
        "recent_renewal": 0,
        "document_type": "passport"
    }

    print("Probability:", model.predict_probability(test_ctx))
    print("Decision:", model.should_show_card(test_ctx))

