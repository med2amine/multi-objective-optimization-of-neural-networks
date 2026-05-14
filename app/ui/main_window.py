from PySide6.QtWidgets import (QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QLabel,QLineEdit,QTextEdit,QPushButton,QFrame)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from app.logic.predictor import predict

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("McLovin")
        self.setMaximumSize(900,600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        left_panel = self._build_left_panel()
        right_panel = self._build_right_panel()

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

    def _build_left_panel(self):
        panel = QFrame()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        title = QLabel("Ticket information")
        title.setFont(QFont("Arial",16,QFont.Bold))
        layout.addWidget(title)

        self.input_titre = self._create_field(layout,"Titre : ")
        self.input_description = self._create_text_field(layout,"Description : ")
        self.input_offre = self._create_field(layout,"Offre : ")
        self.input_demande = self._create_field(layout,"Demande : ")
        self.input_precisez1 = self._create_field(layout,"Precisez (1) : ")
        self.input_precisez2 = self._create_field(layout,"Precisez (2) : ")
        self.input_service = self._create_field(layout,"Service impacté : ")

        self.btn_classify = QPushButton("classer le ticket")
        self.btn_classify.setMinimumHeight(40)
        self.btn_classify.clicked.connect(self._on_classify)
        layout.addWidget(self.btn_classify)

        return panel

    def _build_right_panel(self):
        panel = QFrame()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        title = QLabel("resultats")
        title.setFont(QFont("Arial",16,QFont.Bold))
        layout.addWidget(title)

        layout.addWidget(QLabel("Catégorisation du ticket : "))
        self.result_titre = QLabel("-")
        self.result_titre.setFont(QFont("Arial",12))
        self.result_titre_conf = QLabel("Confidence : -")
        layout.addWidget(self.result_titre)
        layout.addWidget(self.result_titre_conf)

        layout.addSpacing(20)

        layout.addWidget(QLabel("1er niveau : "))
        self.result_1er_niveau = QLabel("-")
        self.result_1er_niveau.setFont(QFont("Arial",12))
        self.result_1er_niveau_conf = QLabel("Confidence : -")
        layout.addWidget(self.result_1er_niveau)
        layout.addWidget(self.result_1er_niveau_conf)

        layout.addStretch()

        return panel
    
    def _create_field(self,layout,label_text):
        layout.addWidget(QLabel(label_text))
        field = QLineEdit()
        layout.addWidget(field)
        return field
    
    def _create_text_field(self,layout,label_text):
        layout.addWidget(QLabel(label_text))
        field = QTextEdit()
        field.setMaximumHeight(100)
        layout.addWidget(field)
        return field
    
    def _on_classify(self):
        inputs = {
            "Titre":                                    self.input_titre.text(),
            "Description":                              self.input_description.toPlainText(),
            "Offre Libellé d'affichage":                self.input_offre.text(),
            "Votre demande concerne Libellé":           self.input_demande.text(),
            "Précisez votre demande (1) Libellé":       self.input_precisez1.text(),
            "Précisez votre demande (2) Libellé":       self.input_precisez2.text(),
            "Service impacté Libellé d'affichage":      self.input_service.text(),
        }

        result = predict(inputs)

        self.result_titre.setText(result["titre"])
        self.result_titre_conf.setText(f"Confidence : {result['titre_confidence']}%")
        self.result_1er_niveau.setText(result["parent"])
        self.result_1er_niveau_conf.setText(f"Confidence : {result['parent_confidence']}%")
