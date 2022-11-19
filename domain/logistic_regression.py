from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from domain.domain import Utility


class MyLog:
    def __init__(self, X_train, X_test, Y_train, Y_test, utility: Utility, max_iter: int = 500):
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.utility = utility
        self.max_iter = max_iter

    def _print(self, model, text: str):
        score = model.score(self.X_test, self.Y_test)
        y_pred = model.predict(self.X_test)
        print("# -------------- Logistic Regression ------------------ #")
        print(text)
        print("Test Score: ", score)
        self.utility.getScore(Y_pred=y_pred, Y_test=self.Y_test, average="weighted")

    def run(self):
        self.multi_newton_l2()
        self.multi_sag_l2()
        self.multi_saga_elastic_1ratio()
        self.multi_saga_elastic_0ratio()
        self.multi_saga_l1()
        self.multi_saga_l2()

    # --- Multinomial, Newton-CG Solver, and l2 penalty --- #
    def multi_newton_l2(self):
        myLog = LogisticRegression(multi_class="multinomial", solver="newton-cg", max_iter=self.max_iter, penalty="l2")
        myLog.fit(self.X_train, self.Y_train)
        self._print(myLog, "# --- Multinomial, Newton-CG Solver, and l2 penalty --- #")

    # --- Multinomial, Sag Solver, and l2 penalty --- #
    def multi_sag_l2(self):
        myLog = LogisticRegression(multi_class="multinomial", solver="sag", max_iter=self.max_iter, penalty="l2")
        myLog.fit(self.X_train, self.Y_train)
        self._print(myLog, "# ------ Multinomial, Sag Solver, and l2 penalty ------ #")

    # --- Multinomial, Saga Solver, and ElasticNet penalty (l1_ratio 1) --- #
    def multi_saga_elastic_1ratio(self):
        myLog = LogisticRegression(multi_class="multinomial", solver="saga", max_iter=self.max_iter,
                                   penalty="elasticnet", l1_ratio=0)
        myLog.fit(self.X_train, self.Y_train)
        self._print(myLog, "# ------ Multinomial, Saga Solver, and Elasticnet penalty (l1_ratio 1) ------ #")

    # --- Multinomial, Saga Solver, and ElasticNet penalty (l1_ratio 1) --- #
    def multi_saga_elastic_0ratio(self):
        myLog = LogisticRegression(multi_class="multinomial", solver="saga", max_iter=self.max_iter,
                                   penalty="elasticnet", l1_ratio=0)
        myLog.fit(self.X_train, self.Y_train)
        self._print(myLog, "# ------ Multinomial, Saga Solver, and Elasticnet penalty (l1_ratio 0) ------ #")

    # --- Multinomial, Saga Solver, and l1 penalty --- #
    def multi_saga_l1(self):
        myLog = LogisticRegression(multi_class="multinomial", solver="saga", max_iter=self.max_iter, penalty="l1")
        myLog.fit(self.X_train, self.Y_train)
        self._print(myLog, "# ------ Multinomial, Saga Solver, and Elasticnet penalty ------ #")

    # --- Multinomial, Saga Solver, and l2 penalty --- #
    def multi_saga_l2(self):
        myLog = LogisticRegression(multi_class="multinomial", solver="saga", max_iter=self.max_iter, penalty="l2")
        myLog.fit(self.X_train, self.Y_train)
        self._print(myLog, "# ------ Multinomial, Saga Solver, and l2 penalty ------ #")

    # --- ovr, Saga Solver, and l2 penalty --- #
    def ovr_saga_l2(self):
        myLog = LogisticRegression(multi_class="multinomial", solver="saga", max_iter=self.max_iter, penalty="l2")
        myLog.fit(self.X_train, self.Y_train)
        self._print(myLog, "# ------ Multinomial, Saga Solver, and l2 penalty ------ #")
