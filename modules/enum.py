from enum import Enum


class SalesBotConversationPurpose(Enum):
    DEMO = "book a demo"
    TRIAL = "setup a trial"
    CONTACTS = "get contacts"


class FileType(Enum):
    PDF_FILE = 1
    URL = 2


class UrlLoadingType(Enum):
    RECOGNITION = 1
    HTML_PARSING = 2


class SalesBotVoiceTone(Enum):
    NEUTRAL = "Neutral"
    FORMAL_AND_PROFESSIONAL = "Formal and Professional"
    CONVERSATIONAL_AND_FRIENDLY = "Conversational and Friendly"
    INSPIRATIONAL_AND_MOTIVATIONAL = "Inspirational and Motivational"
    EMPATHETIC_AND_SUPPORTIVE = "Empathetic and Supportive"
    EDUCATIONAL_AND_INFORMATIVE = "Educational and Informative"


class SalesBotResponseSize(Enum):
    SMALL = 50
    MEDIUM = 250
    LARGE = 500
