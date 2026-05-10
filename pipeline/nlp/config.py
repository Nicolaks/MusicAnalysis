"""
config.py
=========
Constantes globales : chemins, stopwords, DDL des 3 tables.
"""

from pathlib import Path

import nltk
from nltk.corpus import stopwords

# ─────────────────────────────────────────────────────────────────────────────
# Chemins
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH = Path("data/warehouse.duckdb")

# ─────────────────────────────────────────────────────────────────────────────
# NLTK — téléchargement silencieux si absent
# ─────────────────────────────────────────────────────────────────────────────

def ensure_nltk():
    for res in ["punkt", "punkt_tab", "stopwords", "averaged_perceptron_tagger"]:
        try:
            nltk.data.find(f"tokenizers/{res}")
        except LookupError:
            nltk.download(res, quiet=True)


ensure_nltk()

# ─────────────────────────────────────────────────────────────────────────────
# Stopwords
# ─────────────────────────────────────────────────────────────────────────────

STOP_WORDS_FR = set(stopwords.words("french"))
STOP_WORDS_EN = set(stopwords.words("english"))

CUSTOM_STOPWORDS = {
    "ouais", "yeah", "hey", "nan", "hein", "han",
    "plus", "tout", "comme", "fait", "faire",
    "bien", "trop", "quand", "toujours",
    "jamais", "fois", "temps", "maintenant",
    "veux", "faut", "dit", "sais", "peu",
    "deux", "rien", "mal", "vie", "monde",
    "fais", "là", "sans", "si", "tous",
    "va", "vais", "ça", "être", "donc",
    "the", "you", "get", "and",
    "sous", "avant", "vois", "ici",
    "oui", "kho", "toutes", "veut",
    "gros", "non", "tiens", "dire",
    "juste", "vrai", "chez", "quoi",
    "leurs", "parle", "parce", "entre",
    "ans", "comment", "après", "depuis",
    "quoi", "car", "dis", "beaucoup", "encore", "sait",
}

ALL_STOPWORDS = STOP_WORDS_FR | CUSTOM_STOPWORDS | STOP_WORDS_EN


# Lexique émotions — mots-clés par catégorie
EMOTION_LEXICON: dict[str, set[str]] = {

    # ========== ÉMOTIONS PRIMAIRES ==========

    "joie": {
        # --- Expressions générales ---
        "joie", "bonheur", "allégresse", "euphorie", "extase", "ivresse", "ravissement", "jubilation", "exultation", "triomphe",
        "félicité", "beatitude", "émerveillement", "éblouissement", "enchantement", "délire", "folie", "transcendance", "nirvana", "kiff",

        # --- Actions et réactions physiques ---
        "sourire", "rires", "éclats de rire", "fou rire", "rigoler", "délirer", "halluciner", "mort de rire", "plié en deux", "se tordre",
        "sauter", "danser", "bouger", "gesticuler", "taper des mains", "lever les bras", "faire la fête", "teufer", "défouler", "lâcher prise",
        "crier", "hurler", "scander", "chanter", "raper", "sautiller", "taper du pied", "frapper dans ses mains", "embrasser", "sérrer",

        # --- Argots et verlan ---
        "kiffer", "trop bien", "de ouf", "malade", "sick", "c’est le feu", "j’adore", "trop stylé", "génial", "parfait",
        "ouais", "yeah", "oh putain", "waouh", "j’en reviens pas", "c’est la folie", "j’exulte", "je plane", "je suis haut", "je décolle",
        "j’ai le sourire", "j’ai la pêche", "j’ai la banane", "j’ai le seum inversé", "je suis en mode happy", "je suis sur un nuage", "je suis dans ma bulle",
        "c’est ouf", "c’est dingue", "c’est incroyable", "c’est trop", "c’est la hype", "c’est le délire", "c’est le bordel", "c’est le chaos",
        "j’ai le cœur léger", "j’ai l’âme en paix", "je suis en forme", "je suis en feu", "je suis chaud", "je suis motivé", "je suis boosté",

        # --- Contexte social (rap/hip-hop) ---
        "fête", "soirée", "teuf", "after", "concert", "festival", "scène", "public", "foule", "ambiance", "vibe",
        "énergie", "hype", "délire", "folie", "bordel", "chaleur", "communion", "fraternité", "solidarité", "collectif",
        "succès", "victoire", "réussite", "triomphe", "gloire", "fierté", "orgueil", "satisfaction", "accomplissement", "récompense",
        "projet", "rêve", "ambition", "objectif", "but", "désir", "passion", "motivation", "détermination", "volonté",

        # --- Symboles et métaphores ---
        "soleil", "lumière", "ciel", "étoile", "arc-en-ciel", "or", "trésor", "paradis", "rêve éveillé", "nuage",
        "vol", "libre", "lévitation", "apesanteur", "feu d’artifice", "confettis", "champagne", "pétillant", "bulles", "ivresse",
        "feu", "flamme", "braise", "éclair", "foudre", "explosion", "vague", "tsunami", "vent", "brise",
        "cœur", "âme", "esprit", "énergie positive", "ondes", "vibrations", "harmonie", "mélodie", "rythme", "beat",

        # --- Physique et sensations ---
        "frissons", "chair de poule", "cœur qui bat", "souffle coupé", "larmes de joie", "sueur", "adrenaline", "endorphine",
        "sourire aux lèvres", "yeux brillants", "visage rayonnant", "corps léger", "pas dansant", "mouvements fluides", "énergie pure",
        "sensation de plénitude", "impression de tout pouvoir", "sentiment d’invincibilité", "confiance en soi", "puissance",

        # --- Objets et éléments concrets ---
        "micro", "enceinte", "basse", "son", "musique", "beat", "flow", "punchline", "rhyme", "couplet",
        "argents", "thunes", "billets", "cash", "richesse", "luxe", "voiture", "moto", "villa", "bijoux",
        "vêtements", "baskets", "chaîne", "montre", "lunettes", "style", "swag", "look", "tenue", "marque",

        # --- Verlan et argot spécifique ---
        "teuf", "ouam", "reuf", "meuf", "kepon", "ripou", "vrai", "ouf", "chelou", "baltique", "chouette",
        "trop la hype", "c’est le son", "c’est le truc", "c’est le délire", "j’ai le flow", "je suis dans l’game", "je cartonne",
        "je kiffe grave", "je suis à fond", "je suis en mode beast", "je suis un boss", "je suis un roi", "je suis intouchable"
    },

    "tristesse": {
        # --- Expressions générales ---
        "tristesse", "mélancolie", "spleen", "cafard", "blues", "dépine", "désespoir", "abattement", "prostration", "langueur",
        "lassitude", "chagrin", "peine", "souffrance", "douleur", "désolation", "affliction", "désenchantement", "désillusion",
        "pleurs", "larmes", "sanglots", "chialer", "brailler", "gémir", "soupirer", "renifler", "hoqueter", "singulot",

        # --- Actions et réactions ---
        "pleurer", "se recroqueviller", "s’isoler", "se cacher", "fuir", "errer", "marcher seul", "regarder le vide", "fixer le sol",
        "se taire", "murmurer", "écrire", "dessiner", "composer", "chanter triste", "raper sombre", "pleurer en silence", "se rouler en boule",
        "ne plus bouger", "rester au lit", "ne pas sortir", "éviter les gens", "ignorer les appels", "fermer les rideaux", "éteindre la lumière",

        # --- Argots et verlan ---
        "j’ai le seum", "j’ai le blues", "j’ai le cafard", "j’ai la déprime", "j’ai le moral à zéro", "je broie du noir",
        "c’est la loose", "c’est la merde", "c’est l’enfer", "j’ai les boules", "j’ai la rage triste", "je suis à fond le seum",
        "je souffre", "je crève", "je déprime", "je suis au bout", "je tiens plus", "j’ai plus la force", "j’ai plus goût à rien",
        "j’ai le cœur lourd", "j’ai mal", "je suis brisé", "je suis vide", "je suis perdu", "je suis seul", "je suis abandonné",
        "j’ai pété un câble", "j’ai craqué", "je suis en PLS", "je suis en dépression", "je suis au fond du trou", "je touche le fond",

        # --- Contexte social (rap/hip-hop) ---
        "solitude", "isolement", "abandon", "rejet", "exclusion", "manque", "absence", "perte", "deuil", "enterrement",
        "ruine", "échec", "défait", "chômage", "pauvreté", "galère", "misère", "dette", "expulsion", "prison",
        "quartier", "rue", "banlieue", "cité", "ghetto", "bloc", "HLM", "dalle", "bitume", "nuit",
        "trahison", "mensonge", "rupture", "séparation", "mort d’un proche", "disparition", "manque", "nostalgie", "regret", "culpabilité",

        # --- Symboles et métaphores ---
        "nuit", "obscurité", "brouillard", "pluie", "tempête", "hiver", "froid", "glace", "néant", "vide",
        "cœur brisé", "âme en peine", "fantôme", "ombre", "souvenir", "photo", "lettre", "cimetière", "tombe", "adieu",
        "prison", "cage", "enfermement", "chaîne", "menotte", "barreau", "mur", "frontière", "piège", "impasse",
        "cauchemar", "démon", "monstre", "bête noire", "ténèbres", "abîme", "précipice", "chute", "fin",

        # --- Physique et sensations ---
        "cœur serré", "poitrine lourde", "nœud à l’estomac", "maux de tête", "fatigue", "épuisement", "insomnie",
        "cauchemars", "perte d’appétit", "maigrir", "pâleur", "yeux cernés", "regard vide", "voix tremblante", "mains froides",
        "frissons", "sueurs froides", "respiration difficile", "souffle court", "impression d’étouffement", "lourdeur",

        # --- Verlan et argot spécifique ---
        "seum", "teum", "reum", "keum", "chialer", "brailler", "pleurer comme un gosse", "avoir les larmes aux yeux",
        "être en mode zombie", "être dans le noir", "avoir le moral dans les chaussettes", "être à la ramasse", "être en mode survival",
        "avoir la hantise", "avoir les boules", "être en galère", "être dans la merde", "être au bout du rouleau",
        "avoir le cœur en miettes", "être en PLS totale", "avoir la déprime à mort", "être en mode dépressif", "avoir la vie dure"
    },

    "peur": {
        # --- Expressions générales ---
        "peur", "angoisse", "terreur", "effroi", "panique", "crainte", "apprehension", "stress", "anxiété", "trac",
        "vertige", "phobie", "parano", "paranoïa", "psychose", "délire", "cauchemar", "hallucination", "vision", "prémonition",
        "pressentiment", "mauvaise vibe", "mauvais pressentiment", "intuition", "sixième sens", "prédiction", "fatalité",

        # --- Actions et réactions ---
        "trembler", "frissonner", "sursauter", "reculer", "fuir", "se cacher", "se terrer", "se blottir", "se recroqueviller",
        "fermer les yeux", "crier", "hurler", "gémir", "supplier", "prier", "implorer", "regarder derrière soi", "écouter le moindre bruit",
        "retenir son souffle", "se figurer le pire", "anticiper le danger", "éviter les lieux", "changer de trottoir", "se méfier",

        # --- Argots et verlan ---
        "j’ai la trouille", "j’ai les jetons", "j’ai la pétoche", "j’ai la frousse", "j’ai la chiasse", "j’ai la trouille bleue",
        "ça me fait flipper", "ça me fait peur", "j’ai les boules", "je stresse", "je panique", "je déconne pas", "je suis tétanisé",
        "je suis pétrifié", "je suis paralysé", "j’ai envie de gerber", "je vais m’évanouir", "j’ai des sueurs froides", "j’ai le cœur qui bat",
        "j’ai mal au ventre", "je suis en mode parano", "je vois des ombres partout", "je fais des cauchemars éveillés",
        "j’ai la hantise", "j’ai la phobie", "je suis en mode survival", "je suis sur les nerfs", "je suis à cran",

        # --- Contexte social (rap/hip-hop) ---
        "danger", "menace", "piège", "guet-apens", "embuscade", "traque", "chasse", "prédation", "proie", "fuite",
        "rue", "banlieue", "ghetto", "nuit", "ruelle", "immeuble", "cage d’escalier", "parking", "sous-sol", "toit",
        "flic", "keuf", "police", "perquisition", "arrestation", "prison", "bagne", "condamnation", "amende", "délit",
        "violence", "arme", "couteau", "pistolet", "kalach", "AK-47", "guerre", "règlement de comptes", "beef", "clash",

        # --- Symboles et métaphores ---
        "noir", "obscurité", "brouillard", "nuit noire", "cauchemar", "monstre", "démon", "fantôme", "spectre", "ombre",
        "précipice", "abîme", "vide", "chute", "mort", "fin", "disparition", "néant", "enfer", "purgatoire",
        "piège", "filet", "cage", "prison", "chaîne", "menotte", "barreau", "mur", "frontière", "impasse",
        "feu", "explosion", "tonnerre", "orage", "tempête", "tsunami", "déluge", "cataclysme", "apocalypse",

        # --- Physique et sensations ---
        "sueur froide", "mains moites", "cœur qui s’emballe", "respiration rapide", "nausée", "vertige", "tremblements",
        "faiblesse", "jambes en coton", "bouche sèche", "yeux écarquillés", "pupilles dilatées", "muscles tendus",
        "impression d’étouffement", "souffle court", "maux de tête", "douleur à la poitrine", "frissons",

        # --- Verlan et argot spécifique ---
        "flipper", "stresser", "paniquer", "avoir la pétoche", "avoir les jetons", "être en mode parano", "voir des keufs partout",
        "avoir la trouille de sa race", "être en mode fugitif", "avoir la frousse des flics", "être traqué comme un animal",
        "avoir l’impression d’être suivi", "entendre des bruits bizarres", "avoir des visions", "faire des cauchemars",
        "être en mode survival", "avoir la hantise de la taule", "craindre le pire", "anticiper la merde"
    },

    "colère": {
        # --- Expressions générales ---
        "colère", "rage", "fureur", "courroux", "ire", "haine", "rancœur", "amertume", "ressentiment", "vengeance",
        "colère noire", "colère froide", "colère sourde", "explosion", "crise", "accès", "folie", "délire", "frénésie",
        "fureur aveugle", "rage impuissante", "colère rentrée", "colère refoulée", "colère explosive", "colère destructrice",

        # --- Actions et réactions ---
        "crier", "hurler", "gueuler", "brailler", "tempêter", "tonner", "frapper", "cogner", "tabasser", "battre",
        "détruire", "casser", "briser", "saccager", "incendier", "brûler", "démolir", "piétiner", "écraser", "jeter",
        "insulter", "traiter", "cracher", "menacer", "intimider", "provoquer", "défier", "narguer", "humilier", "mépriser",
        "serrer les poings", "grincer des dents", "contracter la mâchoire", "rougir", "blêmir", "trembler", "suer", "haleter",

        # --- Argots et verlan ---
        "j’ai la rage", "j’ai le seum", "j’ai les boules", "j’en ai marre", "j’en peux plus", "ça me saoule", "ça me gonfle",
        "ça me donne la rage", "je pète un câble", "je pète un plomb", "je vois rouge", "je suis à cran", "je suis chaud",
        "je suis en mode guerre", "je vais tout péter", "je vais tout défoncer", "je vais tout niquer", "je vais tout cramer",
        "je vais tout casser", "je suis hors de moi", "je suis incontrôlable", "je suis dangereux", "je suis imprévisible",
        "je suis en mode beast", "je suis un monstre", "je suis une bête", "je suis enragé", "je suis possédé",

        # --- Contexte social (rap/hip-hop) ---
        "injustice", "abus", "oppression", "tyrannie", "dictature", "esclavage", "chaîne", "prison", "cage", "enfermement",
        "système", "état", "gouvernement", "police", "justice", "loi", "règle", "interdit", "censure", "contrôle",
        "quartier", "rue", "banlieue", "ghetto", "cité", "bloc", "HLM", "dalle", "bitume", "nuit",
        "trahison", "mensonge", "tromperie", "manipulation", "hypocrisie", "mépris", "humiliation", "moquerie", "ridicule", "outrage",

        # --- Symboles et métaphores ---
        "feu", "flamme", "braise", "fournaise", "volcan", "explosion", "tonnerre", "orage", "tempête", "tsunami",
        "sang", "rouge", "noir", "sombre", "obscurité", "cauchemar", "démon", "bête", "fauve", "monstre",
        "couteau", "lame", "arme", "pistolet", "kalach", "guerre", "bataille", "combat", "affrontement", "règlement de comptes",
        "piège", "traque", "chasse", "proie", "prédateur", "violence", "douleur", "souffrance", "blessure", "cicatrice",

        # --- Physique et sensations ---
        "poings serrés", "mâchoire contractée", "visage rouge", "veines gonflées", "sueur", "tremblements", "respiration rapide",
        "cœur qui bat", "maux de tête", "nausée", "vertige", "impression d’étouffement", "muscles tendus", "corps en tension",
        "yeux injectés de sang", "regard noir", "voix rauque", "dents serrées", "ongles qui s’enfoncent dans la paume",

        # --- Verlan et argot spécifique ---
        "seum", "rage", "haine", "venin", "pétage de câble", "pétage de plomb", "mode guerre", "mode beast",
        "mode destructeur", "mode incontrôlable", "avoir le sang chaud", "être à fleur de peau", "avoir les nerfs à vif",
        "être prêt à tout péter", "avoir la hache de guerre", "être en mode vendetta", "avoir soif de vengeance",
        "vouloir du sang", "vouloir tout casser", "être en mode noir", "avoir la rage au ventre", "être un volcan prêt à exploser"
    },

    "surprise": {
        # --- Expressions générales ---
        "surprise", "étonnement", "stupéfaction", "sidération", "ébahissement", "saisissement", "interloqué", "médusé",
        "bouche bée", "yeux écarquillés", "incrédulité", "doute", "scepticisme", "méfiance", "questionnement", "perplexité",
        "confusion", "désarroi", "désorientation", "choc", "sursaut", "ahurissement", "consternation", "admiration", "émerveillement",

        # --- Actions et réactions ---
        "sursauter", "reculer", "ouvrir grand les yeux", "lever les sourcils", "bâiller", "haleter", "s’exclamer", "crier",
        "hurler", "glousser", "regarder deux fois", "frotter ses yeux", "se pincer", "vérifier", "chercher", "comprendre",
        "réfléchir", "analyser", "digérer", "assimiler", "resté sans voix", "bégayer", "balbutier", "tousser", "avaler sa salive",

        # --- Argots et verlan ---
        "putain", "waouh", "oh merde", "quoi ?!", "c’est pas vrai ?!", "j’hallucine", "j’en reviens pas", "c’est ouf", "c’est dingue",
        "c’est incroyable", "j’ai pas cru", "j’ai cru rêver", "j’ai cru délirer", "c’est la folie", "c’est le bordel", "c’est le chaos",
        "j’ai le souffle coupé", "j’ai le cœur qui s’arrête", "je suis scotché", "je suis sonné", "je suis choqué", "je suis sous le choc",
        "je déconne pas", "c’est sérieux ?", "t’es sérieux ?", "j’ai capté ?", "j’ai bien entendu ?", "c’est une blague ?",
        "j’ai les yeux qui sortent de la tête", "je tombe des nues", "je suis sur le cul", "je suis K.O.", "je suis sonné",

        # --- Contexte social (rap/hip-hop) ---
        "coup de théâtre", "rebondissement", "retournement", "révélation", "secret", "mystère", "énigme", "surprise",
        "cadeau", "farce", "canular", "poisson d’avril", "mystification", "tromperie", "illusion", "mirage", "rêve", "cauchemar",
        "succès", "échec", "trahison", "réussite", "rencontre", "retrouvailles", "perte", "découverte", "révélation", "scandale",
        "beef", "clash", "diss", "punchline", "couplet", "son", "morceau", "album", "concert", "festival",

        # --- Symboles et métaphores ---
        "éclair", "foudre", "tonnerre", "explosion", "bombe", "coup de massue", "coup de poing", "secousse", "tremblement de terre",
        "tsunami", "vague", "raz-de-marée", "miroir brisé", "monde à l’envers", "réveil brutal", "chute", "vol", "lévitation",
        "rêve éveillé", "hallucination", "illusion", "mystère", "énigme", "casse-tête", "puzzle", "pièce manquante", "découverte",

        # --- Physique et sensations ---
        "frissons", "chair de poule", "sursaut", "respiration bloquée", "cœur qui s’emballe", "mains sur la bouche",
        "yeux grands ouverts", "sourcils levés", "bouche ouverte", "silence", "bégaiement", "toux", "halètement",
        "impression de rêver", "sensation d’irréalité", "étourdissement", "vertige", "faiblesse dans les jambes",

        # --- Verlan et argot spécifique ---
        "ouam", "teuf", "reuf", "meuf", "kepon", "ripou", "vrai", "ouf", "chelou", "baltique", "chouette",
        "c’est le son", "c’est le truc", "c’est le délire", "j’ai capté", "j’ai pigé", "j’ai compris", "j’ai le flow",
        "je suis en mode WTF", "je suis en PLS", "je suis choqué comme jamais", "j’ai l’impression d’être dans un film",
        "c’est trop bizarre", "c’est trop random", "j’ai l’impression de rêver", "je suis en mode ‘putain mais quoi’",
        "j’ai les yeux qui clignotent", "je suis en mode ‘j’y crois pas’", "c’est la claque", "c’est le choc", "c’est l’explosion"
    },

    "dégoût": {
        # --- Expressions générales ---
        "dégoût", "nausée", "écœurement", "haut-le-cœur", "vomissement", "répulsion", "aversion", "horreur", "effroi", "révolte",
        "indignation", "scandale", "choc", "consternation", "désapprobation", "mépris", "haine", "rage", "colère", "fureur",
        "désgout", "répugnance", "antipathie", "hostilité", "rejet", "exécration", "abomination", "abjection", "infamie",

        # --- Actions et réactions ---
        "recracher", "vomir", "cracher", "dégoûter", "repousser", "fuir", "éviter", "détourner le regard", "se boucher le nez",
        "faire la grimace", "insulter", "traiter de", "humilier", "mépriser", "ignorer", "boycotter", "dénoncer", "exposer",
        "montrer du doigt", "juger", "condamner", "critiquer", "moquer", "ridiculiser", "diaboliser", "exécrer", "haïr",

        # --- Argots et verlan ---
        "beurk", "dégoûtant", "immonde", "cradingue", "dégueulasse", "c’est dégueu", "ça me dégoûte", "j’ai la gerbe",
        "j’ai envie de gerber", "ça me donne la nausée", "c’est immonde", "c’est crade", "c’est sale", "c’est pourri",
        "c’est infect", "j’en peux plus", "ça me retourne", "ça me soulève le cœur", "j’ai mal au ventre", "je vais vomir",
        "c’est la honte", "c’est indigne", "c’est répugnant", "c’est ignoble", "c’est abject", "c’est immonde", "c’est dégueulasse",
        "j’ai les haut-le-cœur", "je suis écœuré", "je suis revolté", "je suis indigné", "je suis choqué",

        # --- Contexte social (rap/hip-hop) ---
        "saleté", "crasse", "ordure", "déchet", "immondice", "pourriture", "infection", "maladie", "virus", "bactérie",
        "trahison", "mensonge", "hypocrisie", "manipulation", "corruption", "pourriture", "système", "politique", "police", "injustice",
        "drogue", "deal", "trafic", "came", "poudre", "crack", "héroïne", "overdose", "dépendance", "manque",
        "violence", "meurtre", "assassinat", "sang", "couteau", "pistolet", "guerre", "règlement de comptes", "beef", "clash",

        # --- Symboles et métaphores ---
        "pourriture", "charogne", "cadavre", "excrement", "merde", "vomi", "bave", "sang", "pus", "infection",
        "noir", "sombre", "obscurité", "cauchemar", "enfer", "démon", "monstre", "bête", "rat", "ver",
        "ordure", "immondice", "déchet", "poubelle", "égout", "toilettes", "fumier", "cloaque", "marais", "fange",
        "ténèbres", "abîme", "précipice", "chute", "fin", "disparition", "néant", "vide", "mort",

        # --- Physique et sensations ---
        "nausée", "vomissement", "sueurs froides", "frissons", "maux de ventre", "maux de tête", "vertige", "faiblesse",
        "malaise", "étourdissement", "impression de malaise", "goût amer dans la bouche", "odeurs nauséabondes",
        "respiration difficile", "cœur qui se serre", "estomac qui se retourne", "envie de cracher", "salivation excessive",

        # --- Verlan et argot spécifique ---
        "c’est la cata", "c’est l’horreur", "c’est l’enfer", "c’est le summum du dégoût", "j’ai envie de gerber ma vie",
        "c’est trop crade", "c’est dégueu à mort", "j’ai les boyaux qui se retournent", "je suis dégoûté de tout",
        "c’est la honte absolue", "c’est immonde à pisser", "j’ai l’impression de pourrir", "je suis dans la merde jusqu’au cou",
        "c’est la lie de la société", "c’est la pourriture ambiante", "j’ai envie de tout brûler", "c’est la fin du monde"
    },

    # ========== ÉMOTIONS SECONDAIRES ==========

    "honte": {
        # --- Expressions générales ---
        "honte", "humiliation", "déshonneur", "disgrâce", "opprobre", "infamie", "stigmate", "tache", "souillure", "salissure",
        "culpabilité", "remords", "regret", "repentir", "confession", "aveu", "pardon", "excuse", "réparation", "expier",
        "gêne", "malaise", "embarras", "confusion", "timidité", "pudeur", "réserve", "retrait", "isolement", "solitude",

        # --- Actions et réactions ---
        "rougir", "baisser les yeux", "se cacher le visage", "fuir le regard", "éviter", "se terrer", "se faire discret",
        "disparaître", "s’effacer", "se fondre dans la masse", "pleurer", "sangloter", "gémir", "se morfondre",
        "se ronger les sangs", "se flageller", "s’auto-flageller", "se punir", "se haïr", "se mépriser", "se dévaloriser",
        "mentir", "cacher", "nier", "inventer des excuses", "fuir les responsabilités", "éviter les confrontations",

        # --- Argots et verlan ---
        "j’ai honte", "j’ai la honte", "je suis gêné", "je suis mal", "je me sens sale", "je me sens minable", "je me sens nul",
        "je me sens indigne", "je me sens coupable", "j’ai merdé", "j’ai foiré", "j’ai tout gâché", "j’ai tout raté",
        "j’ai déçu", "j’ai trahi", "j’ai menti", "j’ai volé", "j’ai triché", "j’ai trompé", "j’ai l’impression d’être un moins que rien",
        "j’ai la honte de ma vie", "je suis au fond du trou", "je me sens comme une merde", "je suis dégoûté de moi",
        "je me cache", "je fuis", "je me planque", "je me fais tout petit", "je me sens transparent", "je veux disparaître",

        # --- Contexte social (rap/hip-hop) ---
        "échec", "défait", "ruine", "faillite", "pauvreté", "galère", "misère", "dette", "vol", "escroquerie",
        "trahison", "mensonge", "tromperie", "adultère", "infidélité", "abandon", "rejet", "exclusion", "moquerie", "ridicule",
        "quartier", "rue", "banlieue", "cité", "ghetto", "bloc", "HLM", "famille", "amis", "proches",
        "rumeurs", "ragots", "commérages", "honte publique", "humiliation publique", "scandale", "déshonneur", "exclusion sociale",

        # --- Symboles et métaphores ---
        "tache indélébile", "stigmate", "cicatrice", "blessure", "plie", "marque", "souillure", "saleté", "ordure", "pourriture",
        "nuit", "obscurité", "brouillard", "cauchemar", "enfer", "purgatoire", "jugement", "condamnation", "châtiment", "punition",
        "masque", "faux-semblant", "double vie", "secret", "mensonge", "hypocrisie", "trahison", "lâcheté", "faiblesse", "cowardise",

        # --- Physique et sensations ---
        "visage rouge", "yeux baissés", "posture voûtée", "mains qui tremblent", "voix tremblante", "sueurs froides",
        "maux de ventre", "nausée", "vertige", "faiblesse", "impression d’étouffement", "respiration rapide",
        "cœur qui bat la chamade", "estomac noué", "bouche sèche", "larmes aux yeux", "regard fuyant",

        # --- Verlan et argot spécifique ---
        "j’ai la hontise", "j’ai la teuhon", "je suis en mode honte", "j’ai l’impression d’être un raté",
        "je me sens comme un moins que rien", "j’ai la honte de mon quartier", "j’ai la honte de ma vie", "je me cache des keufs",
        "je fuis les regards", "je me planque comme un lâche", "j’ai l’impression de trahir mes racines",
        "je me sens indigne de mon nom", "j’ai la honte de mes actes", "je veux me racheter", "je veux me faire pardonner",
        "j’ai la honte de ma famille", "je me sens sale comme un traître", "j’ai l’impression d’être un faux jeton"
    },

    "sympathie": {
        # --- Expressions générales ---
        "sympathie", "affection", "attachement", "tendresse", "amitié", "fraternité", "solidarité", "complicité", "empathie",
        "compassion", "bienveillance", "amour", "coup de cœur", "coup de foudre", "admiration", "respect", "estime",
        "confiance", "loyauté", "fidélité", "dévouement", "soutien", "encouragement", "motivation", "inspiration", "guidance",
        "harmonie", "connexion", "alchimie", "fusion", "union", "partage", "générosité", "don", "offrande", "sacrifice",

        # --- Actions et réactions ---
        "soutenir", "aider", "protéger", "défendre", "conseiller", "écouter", "comprendre", "partager", "donner", "offrir",
        "embrasser", "câliner", "serrer dans ses bras", "taper dans le dos", "sourire", "rassurer", "encourager",
        "motiver", "inspirer", "guider", "accompagner", "soutenir moralement", "être là", "prêter main-forte", "tendre la main",
        "féliciter", "applaudir", "célébrer", "fêter", "remercier", "exprimer sa gratitude", "montrer son affection",

        # --- Argots et verlan ---
        "je kiffe", "je t’aime bien", "t’es trop stylé", "t’es un boss", "t’es un ouf", "t’es un vrai", "t’es solide",
        "t’es loyal", "t’es un frère", "t’es une famille", "t’es mon g", "t’es mon pote", "t’es mon frérot",
        "t’es mon frère de cœur", "t’es mon frère de sang", "t’es mon frère d’armes", "t’es mon partenaire", "t’es mon complice",
        "je t’admire", "je te respecte", "t’es mon idole", "t’es mon modèle", "t’es une légende", "t’es un mythe",
        "t’es unique", "t’es irremplaçable", "t’es important pour moi", "je compte sur toi", "t’es mon rocher",
        "t’es mon pilier", "t’es ma force", "t’es mon inspiration", "t’es ma motivation", "t’es mon rayon de soleil",

        # --- Contexte social (rap/hip-hop) ---
        "famille", "amis", "proches", "voisins", "communauté", "groupe", "équipe", "bande", "frères", "sœurs",
        "rencontre", "discussion", "échange", "partage", "souvenir", "mémoire", "histoire", "vécue", "expérience", "aventure",
        "quartier", "rue", "banlieue", "cité", "ghetto", "bloc", "HLM", "dalle", "bitume", "coin de rue",
        "solidarité", "entraide", "soutien mutuel", "résistance", "lutte", "combattant", "guerrier", "soldat", "troupe", "armée",

        # --- Symboles et métaphores ---
        "cœur", "âme", "lien", "chaîne", "pont", "lumière", "chaleur", "feu", "flamme", "étoile",
        "main tendue", "épaule", "rocher", "pilier", "ancrage", "racine", "arbre", "famille", "tribu", "clan",
        "fraternité", "unité", "harmonie", "paix", "amour", "espoir", "rêve", "utopie", "idéal", "futur",

        # --- Physique et sensations ---
        "sourire", "rires", "éclats de rire", "larmes de joie", "frissons", "chair de poule", "cœur qui bat",
        "chaleur", "réconfort", "apaisement", "sérénité", "bien-être", "plénitude", "énergie positive", "ondes positives",
        "impression de sécurité", "sentiment de protection", "confiance", "paix intérieure", "bonheur simple",

        # --- Verlan et argot spécifique ---
        "t’es un vrai G", "t’es un vrai reuf", "t’es un vrai meuf", "t’es un vrai kepon", "t’es un vrai boss",
        "t’es un vrai guerrier", "t’es un vrai soldat", "t’es un vrai frère", "t’es une vraie perle", "t’es un diamant brut",
        "je t’ai dans le cœur", "t’es gravé en moi", "t’es mon sang", "t’es mon âme sœur", "t’es mon double",
        "t’es mon miroir", "t’es mon reflet", "t’es ma moitié", "t’es mon autre moi", "je te kiffe grave"
    },

    "embarras": {
        # --- Expressions générales ---
        "embarras", "gêne", "malaise", "confusion", "timidité", "pudeur", "réserve", "retrait", "hésitation", "indécision",
        "trouble", "désarroi", "perplexité", "inconfort", "anxiété sociale", "complexe", "infériorité", "insécurité", "doute",
        "rougissement", "bégaiement", "balbutiement", "trac", "stress", "panique", "crainte du jugement", "peur du rejet",

        # --- Actions et réactions ---
        "rougir", "baisser les yeux", "bégayer", "balbutier", "tousser", "s’éclaircir la voix", "éviter le regard",
        "se gratter la tête", "se toucher le visage", "jouer avec ses doigts", "regarder ses pieds", "se faire tout petit",
        "changer de sujet", "éluder", "mentir", "inventer une excuse", "fuir la conversation", "quitter la pièce",
        "se cacher", "se planquer", "éviter les situations", "reporter", "annuler", "faire semblant de ne pas voir",

        # --- Argots et verlan ---
        "j’ai la gêne", "j’ai la honte", "je suis mal", "je suis gêné comme un con", "j’ai l’impression d’être transparent",
        "je me sens nul", "je me sens ridicule", "je me sens stupide", "j’ai fait une boulette", "j’ai fait un faux pas",
        "j’ai merdé", "j’ai foiré", "j’ai tout gâché", "j’ai dit une connerie", "j’ai fait une erreur",
        "je suis dans la merde", "je me sens comme un idiot", "j’ai l’impression que tout le monde me regarde",
        "je suis en mode ‘je veux disparaître’", "j’ai envie de me cacher sous une pierre", "je suis trop gêné",
        "c’est la honte", "c’est l’humiliation", "j’ai les boules", "je suis mal à l’aise", "je me sens pas à ma place",

        # --- Contexte social (rap/hip-hop) ---
        "rencontre", "premier rendez-vous", "prise de parole", "scène", "micro", "public", "foule", "caméra",
        "interview", "débat", "discussion", "conflit", "beef", "clash", "diss", "réponse", "réplique", "punchline",
        "quartier", "rue", "banlieue", "cité", "ghetto", "bloc", "HLM", "famille", "amis", "proches",
        "rumeurs", "ragots", "commérages", "honte publique", "humiliation", "moquerie", "ridicule", "rejet",

        # --- Symboles et métaphores ---
        "masque", "faux-semblant", "camouflage", "dissimulation", "secret", "mensonge", "hypocrisie", "trahison",
        "piège", "filet", "cage", "prison", "chaîne", "menotte", "barreau", "mur", "frontière", "obstacle",
        "nuit", "brouillard", "cauchemar", "enfer", "purgatoire", "jugement", "condamnation", "châtiment", "punition",

        # --- Physique et sensations ---
        "visage rouge", "yeux fuyants", "mains moites", "sueurs froides", "cœur qui bat", "respiration rapide",
        "estomac noué", "nausée", "vertige", "faiblesse", "impression d’étouffement", "bouche sèche",
        "tremblements", "frissons", "maux de tête", "maux de ventre", "sensation de malaise",

        # --- Verlan et argot spécifique ---
        "j’ai la teuhon", "j’ai la hontise", "je suis en mode ‘j’ai merdé’", "j’ai l’impression d’être un bouffon",
        "je me sens comme un clown", "j’ai envie de me cacher", "je suis en mode ‘je veux plus sortir’",
        "j’ai la gêne à mort", "je suis trop mal", "j’ai l’impression que tout le monde rit de moi",
        "je suis en mode parano", "j’ai peur du jugement", "je me sens observé", "je me sens jugé", "je me sens exposé"
    },

    "culpabilité": {
        # --- Expressions générales ---
        "culpabilité", "remords", "regret", "repentir", "confession", "aveu", "pardon", "excuse", "réparation", "expier",
        "faute", "erreur", "mauvaise action", "manquement", "violation", "trahison", "mensonge", "tromperie", "vol",
        "dette", "obligation", "devoir", "responsabilité", "honte", "déshonneur", "stigmate", "tache", "souillure",
        "auto-accusation", "auto-flagellation", "auto-punition", "remise en question", "doute", "questionnement", "torture morale",

        # --- Actions et réactions ---
        "pleurer", "sangloter", "gémir", "se morfondre", "se ronger les sangs", "se flageller", "s’auto-flageller",
        "se punir", "se haïr", "se mépriser", "se dévaloriser", "demander pardon", "faire des excuses", "réparer",
        "compenser", "se racheter", "faire amende honorable", "avouer", "confesser", "assumer", "reconnaître ses torts",
        "fuir", "se cacher", "éviter", "ignorer", "nier", "mentir", "inventer des excuses", "trouver des justifications",

        # --- Argots et verlan ---
        "j’ai merdé", "j’ai foiré", "j’ai tout gâché", "j’ai tout raté", "j’ai déçu", "j’ai trahi", "j’ai menti",
        "j’ai volé", "j’ai triché", "j’ai trompé", "j’ai l’impression d’être un moins que rien", "j’ai la culpabilité qui me bouffe",
        "je me sens sale", "je me sens minable", "je me sens indigne", "je me sens coupable", "j’ai les remords",
        "j’ai la conscience lourde", "je suis rongé par la culpabilité", "je me fais des nœuds au cerveau",
        "je rumine", "je broie du noir", "je me sens comme un criminel", "j’ai l’impression d’avoir tué quelqu’un",
        "je veux me faire pardonner", "je veux réparer", "je veux me racheter", "je veux tourner la page",

        # --- Contexte social (rap/hip-hop) ---
        "échec", "défait", "ruine", "faillite", "pauvreté", "galère", "misère", "dette", "vol", "escroquerie",
        "trahison", "mensonge", "tromperie", "adultère", "infidélité", "abandon", "rejet", "exclusion", "moquerie", "ridicule",
        "quartier", "rue", "banlieue", "cité", "ghetto", "bloc", "HLM", "famille", "amis", "proches",
        "rumeurs", "ragots", "commérages", "honte publique", "humiliation publique", "scandale", "déshonneur", "exclusion sociale",

        # --- Symboles et métaphores ---
        "poids", "fardeau", "chaîne", "boulet", "prison", "cage", "enfermement", "ténèbres", "obscurité", "nuit",
        "cauchemar", "démon", "monstre", "bête noire", "fantôme", "spectre", "ombre", "souillure", "tache indélébile",
        "cicatrice", "blessure", "plie", "marque", "stigmate", "péché", "faute", "châtiment", "punition", "jugement",

        # --- Physique et sensations ---
        "cœur lourd", "estomac noué", "nausée", "maux de tête", "vertige", "faiblesse", "impression d’étouffement",
        "respiration difficile", "sueurs froides", "tremblements", "frissons", "maux de ventre", "perte d’appétit",
        "insomnie", "cauchemars", "larmes", "sanglots", "gémissements", "regard vide", "voix tremblante", "mains qui tremblent",

        # --- Verlan et argot spécifique ---
        "j’ai la teuhon", "j’ai la hontise", "je me sens comme un traître", "j’ai l’impression d’être un lâche",
        "je me sens indigne de mon nom", "j’ai la culpabilité au ventre", "je rumine ma culpabilité",
        "je me sens comme un criminel", "j’ai l’impression d’avoir tué mon frère", "je veux me faire pardonner par les miens",
        "je veux réparer mes erreurs", "je veux effacer mes fautes", "je veux me racheter aux yeux de Dieu",
        "je me sens comme un moins que rien", "j’ai la conscience qui me ronge", "je suis en mode ‘je mérite pas le bonheur’"
    },

    "envie": {
        # --- Expressions générales ---
        "envie", "désir", "convoitise", "jalousie", "rivalité", "compétition", "ambition", "soif", "faim", "appétit",
        "attraction", "passion", "obsession", "folie", "délire", "rêve", "aspiration", "volonté", "détermination",
        "motivation", "inspiration", "admiration", "fascination", "attirance", "coup de cœur", "coup de foudre", "envie de posséder",
        "envie d’être", "envie de faire", "envie de réussir", "envie de briller", "envie de dominer", "envie de gagner",

        # --- Actions et réactions ---
        "regarder avec envie", "détailer", "observer", "analyser", "comparer", "se mesurer", "rêver", "imaginer",
        "projetter", "planifier", "travailler", "lutter", "combattre", "se battre", "dépasser", "surpasser", "dominer",
        "vouloir", "désirer", "aspirer", "ambitionner", "viser", "cibler", "conquérir", "obtenir", "posséder", "réussir",
        "copier", "imiter", "s’inspirer", "voler", "usurper", "prendre", "s’approprier", "détourner", "envier", "jalouser",

        # --- Argots et verlan ---
        "j’ai envie", "j’ai trop envie", "j’en rêve", "j’en baise", "j’en kiffe", "j’en ai marre de pas l’avoir",
        "je veux", "je veux trop", "je veux à tout prix", "je veux maintenant", "je veux tout de suite",
        "c’est mon rêve", "c’est mon but", "c’est mon objectif", "c’est ma motivation", "c’est mon obsession",
        "je suis obsédé", "je suis accro", "je suis en mode ‘je veux tout’", "je suis en mode ‘je veux plus’",
        "je suis en mode ‘je mérite mieux’", "je veux sa place", "je veux son style", "je veux son succès", "je veux sa vie",

        # --- Contexte social (rap/hip-hop) ---
        "argent", "thunes", "cash", "pognon", "fric", "billets", "richesse", "luxe", "voiture", "moto",
        "villa", "appart", "bijoux", "chaîne", "montre", "vêtements", "baskets", "marque", "style", "swag",
        "succès", "gloire", "célébrité", "fame", "fans", "public", "scène", "micro", "studio", "concert",
        "femme", "meuf", "reuf", "kepon", "belle", "canon", "bombasse", "déesse", "reine", "princesse",
        "pouvoir", "respect", "reconnaissance", "admiration", "influence", "impact", "héritage", "légende", "mythe",

        # --- Symboles et métaphores ---
        "feu", "flamme", "braise", "fournaise", "volcan", "soif", "faim", "appétit", "désir brûlant", "passion dévorante",
        "rêve", "étoile", "ciel", "horizon", "sommet", "cime", "trône", "couronne", "sceptre", "pouvoir",
        "or", "trésor", "richesse", "abondance", "opulence", "luxe", "brillance", "éclat", "lumière", "gloire",
        "chasse", "proie", "prédateur", "concurrence", "course", "compétition", "bataille", "guerre", "victoire", "triomphe",

        # --- Physique et sensations ---
        "cœur qui bat", "souffle coupé", "frissons", "chair de poule", "yeux brillants", "sourire", "rires",
        "énergie", "adrenaline", "motivation", "détermination", "volonté", "ambition", "passion", "obsession",
        "impression de manquer", "sentiment d’injustice", "colère", "rage", "frustration", "jalousie", "convoitise",

        # --- Verlan et argot spécifique ---
        "j’ai le seum d’envie", "j’ai la rage d’y arriver", "je veux tout niquer pour l’avoir", "je veux tout défoncer pour réussir",
        "j’ai la hargne", "j’ai la niaque", "j’ai la pèche", "j’ai la banane à l’envers", "je veux tout bouffer",
        "je veux tout écraser", "je veux tout dominer", "je veux tout posséder", "je veux être le roi",
        "je veux être le meilleur", "je veux être intouchable", "je veux être une légende", "je veux marquer l’histoire",
        "je veux que tout le monde me kiffe", "je veux que tout le monde me respecte", "je veux être une icône"
    },

    "jalousie": {
        # --- Expressions générales ---
        "jalousie", "envie", "rivalité", "concurrente", "adversaire", "comparaison", "supériorité", "infériorité", "manque",
        "désir", "possession", "amour", "passion", "dévotion", "attachement", "fidélité", "trahison", "infidélité",
        "tromperie", "mensonge", "secret", "doute", "méfiance", "suspicion", "paranoïa", "obsession", "folie",
        "délire", "torture", "souffrance", "douleur", "rage", "colère", "haine", "rancœur", "amertume", "vengeance",

        # --- Actions et réactions ---
        "regarder de travers", "espionner", "surveiller", "contrôler", "posséder", "étouffer", "limiter", "restreindre",
        "interdire", "empêcher", "saboter", "nuire", "détruire", "humilier", "mépriser", "ignorer", "boycotter",
        "comparer", "se mesurer", "rivaliser", "combattre", "défier", "provoquer", "narguer", "se venger",
        "pleurer", "souffrir", "se ronger", "ruminer", "se tourmenter", "se punir", "se haïr", "se mépriser",

        # --- Argots et verlan ---
        "j’ai la jalousie", "j’ai le seum", "j’ai la rage", "j’ai les boules", "j’ai la haine", "je suis vert",
        "je suis noir", "je suis en mode parano", "je stresse", "je panique", "je déconne pas",
        "je veux tout niquer", "je veux tout casser", "je veux tout brûler", "je veux tout détruire",
        "je veux qu’elle souffre", "je veux qu’elle paie", "je veux qu’elle regrette", "je veux qu’elle meure",
        "je suis obsédé", "je suis accro", "je suis en mode ‘je veux tout contrôler’", "je veux tout posséder",
        "je veux être le/la seul(e)", "je veux être le centre de son monde", "je veux qu’elle ne regarde que moi",

        # --- Contexte social (rap/hip-hop) ---
        "amour", "relation", "couple", "meuf", "reuf", "kepon", "ex", "rival", "adversaire", "concurrent",
        "succès", "argent", "gloire", "célébrité", "fame", "fans", "public", "scène", "micro", "studio",
        "style", "swag", "look", "tenue", "marque", "vêtements", "baskets", "bijoux", "chaîne", "montre",
        "voiture", "moto", "villa", "appart", "quartier", "rue", "banlieue", "cité", "ghetto", "bloc",

        # --- Symboles et métaphores ---
        "feu", "flamme", "braise", "fournaise", "volcan", "poison", "venin", "serpent", "vipère", "démon",
        "monstre", "bête", "fauve", "prédateur", "proie", "chasse", "piège", "filet", "cage", "prison",
        "noir", "sombre", "obscurité", "cauchemar", "enfer", "purgatoire", "jugement", "châtiment", "punition",
        "sang", "rouge", "rage", "colère", "haine", "douleur", "souffrance", "blessure", "cicatrice", "torture",

        # --- Physique et sensations ---
        "cœur serré", "estomac noué", "nausée", "maux de tête", "vertige", "faiblesse", "impression d’étouffement",
        "respiration rapide", "sueurs froides", "tremblements", "frissons", "maux de ventre", "larmes",
        "regard noir", "visage fermés", "mâchoire serrée", "poings serrés", "ongles qui s’enfoncent dans la paume",

        # --- Verlan et argot spécifique ---
        "j’ai la teuhon de la jalousie", "je suis en mode ‘je veux tout contrôler’", "je suis en mode ‘personne ne doit l’avoir’",
        "je suis en mode ‘je veux tout pour moi’", "je veux être le/la seul(e) dans son cœur",
        "je veux qu’elle meure de jalousie", "je veux qu’elle souffre comme moi", "je veux qu’elle regrette de m’avoir trahi",
        "je suis en mode ‘je vais tout faire péter’", "je suis en mode ‘je vais tout casser’", "je suis un monstre de jalousie",
        "je suis rongé par la jalousie", "je suis possédé par la jalousie", "je suis en enfer à cause de la jalousie"
    },

    "gratitude": {
        # --- Expressions générales ---
        "gratitude", "reconnaissance", "merci", "remercier", "savoir gré", "appreciation", "estime", "respect", "admiration",
        "fidélité", "loyauté", "dévouement", "amour", "affection", "tendresse", "sympathie", "bienveillance", "générosité",
        "don", "offrande", "cadeau", "partage", "solidarité", "soutien", "aide", "protection", "guidance", "inspiration",

        # --- Actions et réactions ---
        "remercier", "exprimer sa gratitude", "montrer sa reconnaissance", "faire un cadeau", "offrir", "donner",
        "rendre service", "aider", "soutenir", "protéger", "défendre", "honorer", "célébrer", "fêter", "applaudir",
        "sourire", "embrasser", "serrer dans ses bras", "taper dans le dos", "écrire", "appeler", "envoyer un message",
        "prier", "bénir", "reconnaître", "valoriser", "mettre en avant", "citer", "mentionner", "dédier", "hommage",

        # --- Argots et verlan ---
        "merci", "merci beaucoup", "t’es un boss", "t’es un vrai", "t’es solide", "t’es loyal", "t’es un frère",
        "t’es une famille", "t’es mon sauveur", "t’es mon ange", "t’es mon héros", "t’es ma lumière",
        "je te kiffe", "je t’aime bien", "t’es trop stylé", "t’es un ouf", "t’es un reuf", "t’es un kepon",
        "je te dois tout", "je te suis redevable", "je te dois la vie", "je te dois mon succès", "je te dois mon bonheur",
        "je t’en suis éternellement reconnaissant", "je ne t’oublierai jamais", "t’es gravé dans mon cœur", "t’es mon rocher",
        "t’es mon pilier", "t’es ma force", "t’es mon inspiration", "t’es ma motivation", "t’es mon guide",

        # --- Contexte social (rap/hip-hop) ---
        "famille", "amis", "proches", "voisins", "communauté", "groupe", "équipe", "bande", "frères", "sœurs",
        "mentor", "coach", "maître", "professeur", "idole", "modèle", "légende", "mythe", "héro", "sauveur",
        "quartier", "rue", "banlieue", "cité", "ghetto", "bloc", "HLM", "dalle", "bitume", "coin de rue",
        "succès", "réussite", "victoire", "triomphe", "gloire", "fierté", "orgueil", "satisfaction", "accomplissement", "récompense",

        # --- Symboles et métaphores ---
        "cœur", "âme", "lumière", "étoile", "soleil", "feu", "flamme", "chaleur", "réconfort", "paix",
        "main tendue", "câlin", "étreinte", "sourire", "larmes de joie", "rires", "éclats de rire", "bonheur", "plénitude",
        "lien", "chaîne", "pont", "racine", "arbre", "famille", "tribu", "clan", "fraternité", "unité",
        "or", "trésor", "richesse", "abondance", "bénédiction", "miracle", "cadeau du ciel", "destin", "chance",

        # --- Physique et sensations ---
        "sourire", "rires", "éclats de rire", "larmes de joie", "frissons", "chair de poule", "cœur qui bat",
        "chaleur", "réconfort", "apaisement", "sérénité", "bien-être", "plénitude", "énergie positive", "ondes positives",
        "impression de sécurité", "sentiment de protection", "confiance", "paix intérieure", "bonheur simple",

        # --- Verlan et argot spécifique ---
        "t’es un vrai G", "t’es un vrai reuf", "t’es un vrai meuf", "t’es un vrai kepon", "t’es un vrai boss",
        "je te dois tout", "sans toi je serais rien", "t’es mon ange gardien", "t’es mon guide", "t’es ma boussole",
        "t’es mon phare dans la nuit", "t’es mon rayon de soleil", "t’es ma raison de vivre", "t’es mon oxygène",
        "je te kiffe grave", "je t’aime de tout mon cœur", "t’es mon frère pour la vie", "t’es ma famille pour toujours"
    },

    "indignation": {
        # --- Expressions générales ---
        "indignation", "colère", "rage", "fureur", "révolte", "protestation", "dénonciation", "condamnation", "désapprobation",
        "scandale", "choc", "consternation", "horreur", "effroi", "répulsion", "dégoût", "honte", "déshonneur",
        "injustice", "abus", "oppression", "tyrannie", "dictature", "esclavage", "chaîne", "prison", "cage", "enfermement",
        "trahison", "mensonge", "tromperie", "manipulation", "hypocrisie", "corruption", "pourriture", "système", "politique",

        # --- Actions et réactions ---
        "crier", "hurler", "gueuler", "protester", "manifesté", "dénoncer", "exposer", "montrer du doigt", "juger",
        "condamner", "boycotter", "combattre", "lutter", "résister", "se rebeller", "se révolter", "faire la révolution",
        "écrire", "raper", "chanter", "diss", "clash", "punchline", "couplet", "son", "morceau", "album",
        "insulter", "traiter", "humilier", "mépriser", "haïr", "détester", "exécrer", "abhorrer", "maudire",

        # --- Argots et verlan ---
        "j’ai la rage", "j’ai le seum", "j’ai les boules", "j’en ai marre", "j’en peux plus", "ça me saoule",
        "ça me gonfle", "ça me donne la rage", "je pète un câble", "je pète un plomb", "je vois rouge",
        "je suis en mode guerre", "je suis en mode révolution", "je veux tout péter", "je veux tout défoncer",
        "je veux tout niquer", "je veux tout cramer", "je veux tout casser", "je suis hors de moi",
        "c’est la honte", "c’est indigne", "c’est répugnant", "c’est ignoble", "c’est abject", "c’est immonde",
        "j’ai la hantise", "j’ai la nausée", "je suis écœuré", "je suis révolté", "je suis indigné",

        # --- Contexte social (rap/hip-hop) ---
        "quartier", "rue", "banlieue", "cité", "ghetto", "bloc", "HLM", "dalle", "bitume", "nuit",
        "police", "flics", "keufs", "condamnation", "prison", "bagne", "injustice", "système", "état", "gouvernement",
        "pauvreté", "misère", "galère", "chômage", "dette", "expulsion", "ruine", "échec", "défait", "abandon",
        "trahison", "mensonge", "tromperie", "corruption", "pourriture", "violence", "guerre", "règlement de comptes", "beef",

        # --- Symboles et métaphores ---
        "feu", "flamme", "braise", "fournaise", "volcan", "explosion", "tonnerre", "orage", "tempête", "tsunami",
        "sang", "rouge", "noir", "sombre", "obscurité", "cauchemar", "démon", "monstre", "bête", "fauve",
        "chaîne", "menotte", "prison", "cage", "enfermement", "piège", "filet", "traque", "chasse", "proie",
        "justice", "balance", "épée", "bouclier", "combattant", "guerrier", "soldat", "révolutionnaire", "rebelle",

        # --- Physique et sensations ---
        "poings serrés", "mâchoire contractée", "visage rouge", "veines gonflées", "sueur", "tremblements",
        "respiration rapide", "cœur qui bat", "maux de tête", "nausée", "vertige", "impression d’étouffement",
        "muscles tendus", "corps en tension", "yeux injectés de sang", "regard noir", "voix rauque",

        # --- Verlan et argot spécifique ---
        "j’ai la teuhon de l’injustice", "je suis en mode ‘je veux tout faire péter’", "je suis en mode ‘révolution’",
        "je veux tout brûler", "je veux tout détruire", "je veux que ça change", "je veux la justice",
        "je veux que les coupables paient", "je veux que le système tombe", "je veux que la vérité éclate",
        "je suis en mode ‘plus jamais ça’", "je suis en mode ‘jamais sans rien dire’", "je suis un soldat de la cause"
    },

    "mépris": {
        # --- Expressions générales ---
        "mépris", "dédain", "désintérêt", "indifférence", "froideur", "distance", "rejet", "humiliation", "moquerie", "ridicule",
        "haine", "rage", "colère", "rancœur", "amertume", "ressentiment", "vengeance", "punition", "châtiment",
        "arrogance", "supériorité", "infériorité", "complexe", "jalousie", "envie", "rivalité", "concurrente", "adversaire",
        "trahison", "mensonge", "tromperie", "manipulation", "hypocrisie", "corruption", "pourriture", "système", "injustice",

        # --- Actions et réactions ---
        "ignorer", "bouder", "éviter", "fuir", "tourner le dos", "détourner le regard", "faire la sourde oreille",
        "insulter", "traiter", "humilier", "rabaisser", "dévaloriser", "mépriser", "dénigrer", "diffamer",
        "moquer", "se moquer", "ridiculiser", "railler", "sarcasme", "ironie", "cynisme", "moquerie", "persiflage",
        "exclure", "rejeter", "ostraciser", "boycotter", "bannir", "chasser", "éliminer", "détruire", "anéantir",

        # --- Argots et verlan ---
        "j’ai le mépris", "j’ai la haine", "je le/la kiffe pas", "je le/la supporte pas", "je le/la déteste",
        "je le/la méprise", "je le/la méprise de haut", "je le/la regarde de haut", "je le/la considère comme rien",
        "c’est un moins que rien", "c’est un bon à rien", "c’est un raté", "c’est un loosers", "c’est un bouffon",
        "je lui crache dessus", "je lui pisse dessus", "je le/la nique", "je le/la défonce", "je le/la démolis",
        "je le/la réduis en miettes", "je le/la traîne dans la boue", "je le/la salis", "je le/la dévalorise",
        "je suis au-dessus de ça", "je suis trop bien pour lui/elle", "je vaux mieux que lui/elle",

        # --- Contexte social (rap/hip-hop) ---
        "quartier", "rue", "banlieue", "cité", "ghetto", "bloc", "HLM", "dalle", "bitume", "nuit",
        "flic", "keuf", "police", "justice", "système", "état", "gouvernement", "politique", "corruption",
        "traître", "mouchard", "balance", "indicateurs", "collaborateur", "lâche", "faux jeton", "hypocrite", "menteur",
        "pauvre", "riche", "bourgeois", "nanti", "privilégié", "fils à papa", "arrogant", "prétentieux", "vaniteux",

        # --- Symboles et métaphores ---
        "boue", "crasse", "ordure", "déchet", "immondice", "pourriture", "infection", "maladie", "virus", "bactérie",
        "noir", "sombre", "obscurité", "cauchemar", "enfer", "démon", "monstre", "bête", "rat", "ver",
        "poussière", "saleté", "souillure", "tache", "stigmate", "cicatrice", "blessure", "honte", "déshonneur",
        "chaîne", "menotte", "prison", "cage", "enfermement", "piège", "filet", "traque", "chasse", "proie",

        # --- Physique et sensations ---
        "lèvre retroussée", "nez plissé", "regard méprisant", "sourire narquois", "ricanement", "haussement d’épaules",
        "souffle bruyant", "geste de dédain", "posture hautaine", "voix froide", "ton sec", "silence pesant",
        "impression de supériorité", "sentiment de dégoût", "nausée", "maux d’estomac", "frissons",

        # --- Verlan et argot spécifique ---
        "j’ai le teum de mépris", "je le/la kiffe grave pas", "je le/la méprise à mort", "je le/la considère comme de la merde",
        "je lui crache au visage", "je lui pisse à la raie", "je le/la traîne dans la boue", "je le/la réduis en poussière",
        "je suis au-dessus de sa race", "je vaux 1000 fois mieux que lui/elle", "je le/la regarde même pas",
        "je fais comme si elle existe pas", "je le/la nique de la tête aux pieds"
    },

    "espoir": {
        # --- Expressions générales ---
        "espoir", "rêve", "avenir", "demain", "futur", "lendemain", "horizon", "chemin", "route", "voyage",
        "foi", "croyance", "confiance", "optimisme", "perspective", "projet", "but", "objectif", "ambition",
        "désir", "volonté", "détermination", "courage", "ténacité", "persévérance", "combattant", "résistance", "lutte",
        "renaissance", "renouveau", "changement", "évolution", "progrès", "métamorphose", "transformation", "grandir", "apprendre",

        # --- Actions et réactions ---
        "rêver", "imaginer", "visualiser", "projetter", "planifier", "anticiper", "préparer", "travailler", "lutter", "combattre",
        "espérer", "croire", "avoir foi", "prier", "méditer", "se battre", "persévérer", "ne pas abandonner", "continuer",
        "se relever", "recommencer", "repartir", "renaître", "se reconstruire", "se réinventer", "changer", "évoluer", "grandir",
        "chercher", "trouver", "découvrir", "innover", "créer", "inventer", "construire", "bâtir", "édifier", "réussir",

        # --- Argots et verlan ---
        "j’ai l’espoir", "j’ai la foi", "je crois en demain", "je crois en l’avenir", "je crois en moi",
        "je crois en mon rêve", "je crois en mon projet", "je crois en ma réussite", "je crois en mon succès",
        "je vais y arriver", "je vais réussir", "je vais percer", "je vais cartonner", "je vais exploser",
        "je vais tout défoncer", "je vais tout niquer", "je vais tout écraser", "je vais tout dominer",
        "je suis motivé", "je suis boosté", "je suis en mode ‘je lâche rien’", "je suis en mode ‘je vais tout avoir’",
        "je suis en mode ‘le meilleur arrive’", "je suis en mode ‘demain sera meilleur’", "je suis en mode ‘je vais tout changer’",

        # --- Contexte social (rap/hip-hop) ---
        "quartier", "rue", "banlieue", "cité", "ghetto", "bloc", "HLM", "dalle", "bitume", "nuit",
        "famille", "amis", "proches", "communauté", "groupe", "équipe", "bande", "frères", "sœurs", "enfants",
        "succès", "réussite", "victoire", "triomphe", "gloire", "fierté", "orgueil", "satisfaction", "accomplissement", "récompense",
        "argent", "thunes", "cash", "pognon", "fric", "billets", "richesse", "luxe", "voiture", "villa",

        # --- Symboles et métaphores ---
        "lumière", "aube", "lever de soleil", "étoile", "ciel", "horizon", "chemin", "route", "voyage", "dépassement",
        "graine", "arbre", "fleur", "bourgeon", "renaissance", "printemps", "jeunesse", "futur", "destin", "chance",
        "feu", "flamme", "braise", "étincelle", "énergie", "motivation", "passion", "désir", "volonté", "courage",
        "oiseau", "vol", "ailes", "libre", "lévitation", "apesanteur", "rêve éveillé", "utopie", "paradis", "ciel",

        # --- Physique et sensations ---
        "cœur qui bat", "souffle léger", "frissons", "chair de poule", "yeux brillants", "sourire", "rires",
        "énergie", "adrenaline", "motivation", "détermination", "volonté", "ambition", "passion", "obsession",
        "impression de légèreté", "sentiment de liberté", "confiance", "paix intérieure", "bonheur simple",

        # --- Verlan et argot spécifique ---
        "j’ai la hype de l’espoir", "je kiffe mon avenir", "je suis en mode ‘demain sera meilleur’",
        "je suis en mode ‘je vais tout défoncer’", "je suis en mode ‘je vais percer’", "je suis en mode ‘je vais cartonner’",
        "je suis en mode ‘le meilleur est à venir’", "je suis en mode ‘je lâche rien’", "je suis en mode ‘je vais tout avoir’",
        "je suis un guerrier de l’espoir", "je suis un combattant du rêve", "je suis un soldat de la réussite"
    },

    "désespoir": {
        # --- Expressions générales ---
        "désespoir", "abattement", "découragement", "résignation", "renoncement", "abandon", "défait", "échec", "ruine",
        "effondrement", "chute", "vide", "néant", "obscurité", "nuit", "brouillard", "froid", "glace", "désert",
        "pleurs", "larmes", "souffrance", "douleur", "chagrin", "peine", "torture", "agonie", "supplice", "martyre",
        "inutilité", "vanité", "absurdité", "sens", "désillusion", "désenchantement", "trahison", "mensonge", "tromperie",

        # --- Actions et réactions ---
        "pleurer", "sangloter", "gémir", "se morfondre", "se ronger les sangs", "se flageller", "s’auto-flageller",
        "abandonner", "renoncer", "lâcher prise", "baisser les bras", "se résigner", "se décourager", "perdre espoir",
        "se recroqueviller", "s’isoler", "se cacher", "fuir", "errer", "marcher sans but", "regarder le vide", "fixer le sol",
        "ne plus bouger", "rester au lit", "ne pas sortir", "éviter les gens", "ignorer les appels", "fermer les rideaux",

        # --- Argots et verlan ---
        "j’ai le seum", "j’ai le blues", "j’ai le cafard", "j’ai la déprime", "j’ai le moral à zéro",
        "je broie du noir", "j’en peux plus", "je tiens plus", "j’ai plus la force", "j’ai plus goût à rien",
        "c’est la loose", "c’est la merde", "c’est l’enfer", "j’ai les boules", "j’ai la rage triste",
        "je suis au bout", "je suis au fond du trou", "je touche le fond", "je suis en PLS", "je suis en dépression",
        "j’ai l’impression que tout est fini", "j’ai l’impression que rien ne changera", "j’ai l’impression d’être maudit",
        "je suis en mode ‘à quoi bon’", "je suis en mode ‘plus rien n’a de sens’", "je suis en mode ‘je veux mourir’",

        # --- Contexte social (rap/hip-hop) ---
        "solitude", "isolement", "abandon", "rejet", "exclusion", "manque", "absence", "perte", "deuil", "enterrement",
        "ruine", "échec", "défait", "chômage", "pauvreté", "galère", "misère", "dette", "expulsion", "prison",
        "quartier", "rue", "banlieue", "cité", "ghetto", "bloc", "HLM", "dalle", "bitume", "nuit",
        "trahison", "mensonge", "rupture", "séparation", "mort d’un proche", "disparition", "manque", "nostalgie",

        # --- Symboles et métaphores ---
        "nuit", "obscurité", "brouillard", "pluie", "tempête", "hiver", "froid", "glace", "néant", "vide",
        "cœur brisé", "âme en peine", "fantôme", "ombre", "souvenir", "photo", "lettre", "cimetière", "tombe", "adieu",
        "prison", "cage", "enfermement", "chaîne", "menotte", "barreau", "mur", "frontière", "piège", "impasse",
        "cauchemar", "démon", "monstre", "bête noire", "ténèbres", "abîme", "précipice", "chute", "fin",

        # --- Physique et sensations ---
        "cœur serré", "poitrine lourde", "nœud à l’estomac", "maux de tête", "fatigue", "épuisement", "insomnie",
        "cauchemars", "perte d’appétit", "maigrir", "pâleur", "yeux cernés", "regard vide", "voix tremblante", "mains froides",
        "frissons", "sueurs froides", "respiration difficile", "souffle court", "impression d’étouffement", "lourdeur",

        # --- Verlan et argot spécifique ---
        "j’ai la teuhon du désespoir", "je suis en mode ‘j’ai plus d’espoir’", "je suis en mode ‘tout est fini’",
        "je suis en mode ‘je veux plus vivre’", "je suis en mode ‘je veux disparaître’", "je suis un fantôme",
        "je suis un mort-vivant", "je suis en enfer sur Terre", "je suis maudit", "je suis un damné",
        "je suis en mode ‘à quoi bon continuer’", "je suis en mode ‘personne me comprend’", "je suis en mode ‘je suis seul au monde’"
    },

    "méfiance": {
        # --- Expressions générales ---
        "méfiance", "suspicion", "doute", "incrédulité", "scepticisme", "crainte", "apprehension", "prudence", "circonspection",
        "paranoïa", "parano", "délire", "hallucination", "vision", "prémonition", "pressentiment", "mauvaise vibe",
        "intuition", "sixième sens", "instinct", "flair", "précaution", "vigilance", "attention", "surveillance", "contrôle",

        # --- Actions et réactions ---
        "observer", "surveiller", "guetter", "épier", "espionner", "regarder de travers", "écouter aux aguets",
        "poser des questions", "vérifier", "contrôler", "douter", "remettre en question", "se méfier", "se méfier de tout",
        "éviter", "fuir", "se cacher", "se protéger", "se défendre", "anticiper", "prévoir", "se préparer",
        "mentir", "tromper", "manipuler", "jouer un rôle", "faire semblant", "dissimuler", "cacher", "masquer",

        # --- Argots et verlan ---
        "j’ai la méfiance", "je suis méfiant", "je me méfie", "je me méfie de tout le monde", "je fais gaffe",
        "je suis en mode parano", "je vois des traîtres partout", "je vois des keufs partout", "je vois des balances partout",
        "je suis en mode ‘je fais confiance à personne’", "je suis en mode ‘tout le monde veut me niquer’",
        "je suis en mode ‘je dois tout contrôler’", "je suis en mode ‘je dois tout vérifier’", "je suis en mode ‘survie’",
        "j’ai l’impression d’être traqué", "j’ai l’impression d’être espionné", "j’ai l’impression qu’on me ment",
        "je suis en mode ‘je crois personne’", "je suis en mode ‘je vérifie tout’", "je suis un loup solitaire",

        # --- Contexte social (rap/hip-hop) ---
        "quartier", "rue", "banlieue", "cité", "ghetto", "bloc", "HLM", "dalle", "bitume", "nuit",
        "flic", "keuf", "police", "indicateurs", "mouchard", "balance", "traître", "hypocrite", "menteur", "faux jeton",
        "drogue", "deal", "trafic", "came", "poudre", "crack", "héroïne", "overdose", "dépendance", "manque",
        "argent", "thunes", "cash", "pognon", "fric", "billets", "richesse", "luxe", "voiture", "villa",

        # --- Symboles et métaphores ---
        "piège", "filet", "cage", "prison", "chaîne", "menotte", "barreau", "mur", "frontière", "obstacle",
        "nuit", "obscurité", "brouillard", "cauchemar", "enfer", "démon", "monstre", "bête", "fauve", "prédateur",
        "masque", "faux-semblant", "camouflage", "dissimulation", "secret", "mensonge", "hypocrisie", "trahison",
        "loupe", "microscope", "détective", "enquêteur", "espion", "soldat", "guerrier", "combattant", "résistant",

        # --- Physique et sensations ---
        "yeux plissés", "regard méfiant", "sourcils froncés", "posture tendue", "muscles contractés", "respiration rapide",
        "cœur qui bat", "mains moites", "sueurs froides", "frissons", "impression d’être observé", "sentiment de danger",
        "nausée", "maux de tête", "vertige", "faiblesse", "impression d’étouffement", "souffle court",

        # --- Verlan et argot spécifique ---
        "j’ai la teuhon de la méfiance", "je suis en mode ‘je fais gaffe à tout’", "je suis en mode ‘je crois personne’",
        "je suis en mode ‘tout le monde est un ennemi’", "je suis en mode ‘je dois survivre’",
        "je suis un loup dans la bergerie", "je suis un fantôme dans la nuit", "je suis un ombre dans la lumière",
        "je me méfie même de mon ombre", "je vérifie même mes proches", "je suis en mode ‘parano totale’"
    }
}

LEXICAL_FIELDS: dict[str, set[str]] = {
    # --- Thèmes centraux ---
    "argent": {
        # Argent (liquide, gain, perte)
        "thunes", "cash", "oseille", "pognon", "fric", "billets", "liquide", "monnaie", "pièces", "euros",
        "dollars", "millions", "mille", "cent", "dix", "vingt", "cinquante", "centaine", "paquet", "liasse",
        "fortune", "richesse", "pactole", "jackpot", "trésor", "or", "diamant", "bijoux", "rolex", "chaîne",
        # Statut financier
        "riche", "bourré", "plein aux as", "noyé", "blindé", "kiffé", "pauvre", "fauché", "ruiné", "dette",
        "crédit", "prêt", "emprunt", "rembourser", "payer", "facture", "loyer", "impôts", "taxes", "amende",
        # Sources de revenus
        "boulot", "taf", "business", "deal", "vente", "traffic", "combines", "arnaque", "escroquerie", "vol",
        "braquage", "hold-up", "cambriolage", "pillage", "rançon", "parrain", "mafia", "network", "empire", "royaume",
        # Dépenses et style de vie
        "dépenser", "claque", "flamber", "kiffer", "gâter", "offrir", "cadeau", "luxe", "voiture", "moto",
        "villa", "appart", "palace", "hôtel", "voyage", "avion", "bateau", "yacht", "fête", "soirée",
        # Symboles
        "portefeuille", "compte", "banque", "coffre", "sac", "poche", "billet", "monnayeur", "distributeur", "virement"
    },

    "rue": {
        # Lieux
        "quartier", "rue", "banlieue", "cité", "dalle", "bitume", "bloc", "tour", "HLM", "barre",
        "ghetto", "projet", "cage", "cage d’escalier", "hall", "ascenseur", "parking", "terrain", "cour", "immeuble",
        # Éléments urbains
        "trottoir", "caniveau", "graffiti", "tag", "mur", "brique", "béton", "acier", "grillage", "portail",
        "banc", "lampe", "réverbère", "feu", "stop", "panneau", "sens interdit", "rue sans issue", "impasse", "carrefour",
        # Ambiance
        "bruyant", "vivant", "chaud", "froid", "sombre", "dangereux", "calme", "désert", "vide", "plein",
        "foule", "voisins", "voisinage", "communauté", "bande", "groupe", "mec", "gars", "jeune", "vieux",
        # Actions
        "descendre", "monter", "traîner", "errer", "marcher", "courir", "fuir", "se cacher", "observer", "guetter",
        "parler", "discuter", "raconter", "écouter", "regarder", "voir", "savoir", "connaître", "reconnaître", "oublié",
        # Symboles
        "clé", "porte", "serrure", "code", "interphone", "sonnette", "fenêtre", "balcon", "toit", "ciel"
    },

    "famille": {
        # Membres directs
        "mère", "maman", "daronne", "vieille", "père", "papa", "daron", "vieux", "frère", "frérot", "bro",
        "sœur", "sœurcie", "fille", "fils", "enfant", "gamin", "môme", "bébé", "ado", "jeune",
        # Famille élargie
        "oncle", "tonton", "tatie", "cousin", "cousine", "grand-père", "papi", "grand-mère", "mamie", "neveu",
        "nièce", "parrain", "marraine", "belle-mère", "beau-père", "belle-sœur", "beau-frère", "fiancé", "fiancée", "mari",
        # Liens
        "sang", "racines", "origines", "héritage", "nom", "prénom", "histoire", "mémoire", "souvenir", "photo",
        "arbre généalogique", "tradition", "culture", "pays", "ville", "quartier", "maison", "foyer", "chez soi", "nid",
        # Rôles et émotions
        "amour", "affection", "soutien", "protection", "éducation", "conseil", "fierté", "honneur", "respect", "devoir",
        "sacrifice", "travail", "fatigue", "peine", "joie", "bonheur", "tristesse", "manque", "absence", "deuil",
        # Conflits
        "dispute", "conflit", "tension", "colère", "rage", "haine", "trahison", "mensonge", "secret", "pardon", "réconciliation"
    },

    "drogue": {
        # Substances
        "shit", "weed", "herbe", "ganja", "marijuana", "cannabis", "hasch", "haschisch", "résine", "beuh",
        "kiff", "joint", "pétard", "clope", "cigarette", "blunt", "spliff", "oil", "huile", "space",
        "came", "poudre", "blanche", "neige", "cocaïne", "coke", "crack", "caillou", "base", "freebase",
        "ecstasy", "MD", "pilule", "cachet", "acide", "LSD", "champignon", "psilo", "DMT", "ketamine",
        # Consommation
        "fumer", "tirer", "allumer", "aspirer", "sniffer", "renifler", "inhaler", "avaler", "boire", "shooter",
        "défonce", "planer", "trip", "voyage", "hallucination", "délire", "parano", "bad trip", "overdose", "manque",
        # Vente et réseau
        "dealer", "vendeur", "fournisseur", "gros", "détail", "gramme", "once", "kilo", "ballon", "sachet",
        "planque", "stash", "cache", "coffre", "sac", "poche", "deal", "transaction", "échange", "argent",
        # Effets
        "high", "stone", "défonce", "euphorie", "bonheur", "rires", "détente", "calme", "sommeil", "faim",
        "paranoïa", "angoisse", "stress", "tremblement", "sueurs", "maux", "nausée", "vertige", "perte de contrôle", "accoutumance",
        # Répression
        "flic", "police", "perquisition", "arrestation", "prison", "condamnation", "amende", "délit", "trafic", "peine", "cavale"
    },

    "célébrité": {
        # Statut
        "gloire", "succès", "réussite", "star", "vedette", "légende", "icône", "mythe", "idole", "modèle",
        "héro", "roi", "empereur", "dieu", "divin", "immortel", "intouchable", "incontournable", "indétrônable", "unique",
        # Univers du spectacle
        "scène", "micro", "spotlight", "projecteur", "lumières", "public", "foule", "fans", "groupies", "admirateurs",
        "concert", "tournée", "festival", "plateau", "studio", "enregistrement", "clip", "vidéo", "interview", "médias",
        # Récompenses
        "prix", "trophée", "grammy", "victoire", "nomination", "classement", "n°1", "hit", "tube", "carton",
        "disque d’or", "disque de platine", "ventes", "streaming", "vues", "likes", "partages", "tendance", "viral", "buzz",
        # Style de vie
        "luxe", "VIP", "backstage", "loge", "gardes du corps", "sécurité", "voiture avec chauffeur", "jet privé", "hôtel 5*", "soirées",
        # Pressions
        "pression", "attente", "critique", "haine", "jalousie", "envie", "rumeurs", "scandale", "polémique", "procès",
        # Symboles
        "signature", "autographe", "poster", "affiche", "magazine", "une", "couverture", "reportage", "documentaire", "biopic"
    },

    "spiritualité": {
        # Divinités et textes sacrés
        "dieu", "allah", "yahvé", "jésus", "bouddha", "prophète", "saint", "ange", "démon", "esprit",
        "coran", "bible", "torah", "évangile", "sourate", "verset", "prière", "sourate", "hadith", "parabole",
        # Pratiques
        "foi", "croyance", "religion", "islam", "christianisme", "judaïsme", "bouddhisme", "hindouisme", "athéisme", "agnosticisme",
        "méditation", "contemplation", "recueillement", "silence", "jeûne", "pèlerinage", "mosquée", "église", "temple", "synagogue",
        # Destin et morale
        "destin", "fatalité", "karma", "sort", "chance", "miracle", "bénédiction", "malédiction", "épreuve", "souffrance",
        "pardon", "rédemption", "salut", "paradis", "enfer", "jugement", "âme", "esprit", "conscience", "repentir",
        # Symboles
        "croix", "croissant", "étoile de david", "om", "mandala", "rosace", "icône", "relique", "chandelle", "encens",
        # Métaphores
        "lumière", "obscurité", "chemin", "guide", "vérité", "sagesse", "paix", "amour", "espoir", "doute"
    },

    "amour_perdu": {
        # Rupture
        "trahison", "mensonge", "tromperie", "infidélité", "adultère", "cocu", "cornu", "partir", "quitter", "abandonner",
        "rupture", "séparation", "divorce", "adieu", "au revoir", "fin", "terminaison", "échec", "désillusion", "désenchantement",
        # Émotions
        "pleurer", "larmes", "chagrin", "peine", "souffrance", "douleur", "cœur brisé", "blessure", "cicatrice", "manque",
        "regret", "nostalgie", "souvenir", "photo", "lettre", "cadeau", "objet", "parfum", "musique", "chanson",
        # Solitude
        "seul", "isolement", "abandon", "vide", "néant", "silence", "nuit", "obscurité", "brouillard", "pluie",
        # Colère et vengeance
        "haine", "rage", "colère", "vengeance", "punition", "châtiment", "rancœur", "amertume", "ressentiment", "jalousie",
        # Réconciliation (rare)
        "pardon", "excuse", "retour", "réconciliation", "espoir", "recommencer", "oubli", "tourner la page", "nouveau départ", "renaissance"
    },

    "violence": {
        # Armes
        "couteau", "lame", "surin", "dague", "poignard", "hache", "machette", "barre de fer", "chaîne", "bâton",
        "arme", "pistolet", "flingue", "revolver", "kalach", "AK-47", "Uzi", "mitraillette", "fusil", "balle",
        # Actions violentes
        "frapper", "cogner", "tabasser", "battre", "rouer de coups", "gifler", "donner un coup", "poing", "coup de pied", "coup de poing",
        "blesser", "saigner", "couper", "égorger", "tuer", "assassiner", "exécuter", "descendre", "buter", "éliminer",
        # Conflits
        "guerre", "bataille", "combat", "affrontement", "règlement de comptes", "vendetta", "conflit", "tension", "clash", "beef",
        "ennemi", "adversaire", "rival", "opposant", "gang", "bande", "clan", "tribu", "territoire", "quartier",
        # Conséquences
        "sang", "hémorragie", "blessure", "cicatrice", "douleur", "souffrance", "agonie", "mort", "cadavre", "enterrement",
        # Contexte
        "rue", "banlieue", "ghetto", "prison", "bagne", "cellule", "cachot", "police", "flic", "keuf",
        # Métaphores
        "feu", "flamme", "explosion", "tonnerre", "orage", "tempête", "tsunami", "déluge", "chaos", "apocalypse"
    },

    # --- Thèmes complémentaires ---
    "succès": {
        "réussite", "victoire", "triomphe", "gloire", "honneur", "respect", "reconnaissance", "légende", "mythe", "histoire",
        "ascension", "montée", "progrès", "évolution", "métamorphose", "transformation", "renaissance", "renouveau", "réinvention", "comeback",
        "travail", "effort", "sueur", "sang", "larmes", "sacrifice", "persévérance", "détermination", "volonté", "ambition",
        "objectif", "but", "rêve", "vision", "projet", "plan", "stratégie", "méthode", "discipline", "rigueur",
        "talent", "don", "génie", "inspiration", "créativité", "originalité", "style", "flow", "punchline", "métaphore",
        "public", "fans", "admirateurs", "communauté", "influence", "impact", "héritage", "marque", "empreinte", "trace",
        "argent", "richesse", "fortune", "luxe", "pouvoir", "contrôle", "liberté", "indépendance", "autonomie", "maîtrise",
        "médias", "presse", "interviews", "une", "couverture", "titre", "article", "reportage", "documentaire", "biopic",
        "récompense", "prix", "trophée", "grammy", "disque d’or", "disque de platine", "n°1", "hit", "tube", "classement"
    },

    "échec": {
        "défait", "chute", "effondrement", "ruine", "fiasco", "désastre", "catastrophe", "naufrage", "échec", "faux pas",
        "erreur", "faute", "mauvaise décision", "choix", "regret", "remords", "culpabilité", "honte", "doute", "incertitude",
        "perte", "manque", "absence", "disparition", "fin", "terminaison", "abandon", "renoncement", "désistement", "démission",
        "pauvreté", "fauché", "ruiné", "dette", "crédit", "prêt", "emprunt", "saisie", "expulsion", "exclusion",
        "solitude", "isolement", "abandonné", "oublié", "invisible", "méprisé", "moqué", "ridiculisé", "humilié", "trahi",
        "maladie", "souffrance", "douleur", "désespoir", "dépression", "burn-out", "épuisement", "fatigue", "lassitude", "découragement",
        "trahison", "mensonge", "tromperie", "manipulation", "arnaque", "escroquerie", "vol", "braquage", "prison", "condamnation",
        "critique", "moquerie", "ridicule", "haine", "jalousie", "envie", "rumeur", "scandale", "polémique", "disgrâce"
    },

    "liberté": {
        # Concepts
        "libération", "émancipation", "affranchissement", "indépendance", "autonomie", "souveraineté", "maîtrise", "contrôle", "choix", "décision",
        # Symboles
        "vol", "oiseau", "aile", "ciel", "horizon", "vent", "vague", "océan", "route", "chemin",
        # Actions
        "s’échapper", "fuir", "partir", "quitter", "abandonner", "libérer", "délivrer", "sauver", "protéger", "défendre",
        # Contexte
        "prison", "cage", "enfermement", "chaîne", "menotte", "barreau", "mur", "frontière", "limite", "obstacle",
        # Émotions
        "joie", "bonheur", "soulagement", "paix", "sérénité", "espoir", "rêve", "utopie", "idéal", "futur",
        # Métaphores
        "lumière", "aube", "renaissance", "renouveau", "printemps", "fleur", "bourgeon", "graine", "arbre", "racine",
        # Outils
        "clé", "serrure", "porte", "fenêtre", "passage", "pont", "échelle", "corde", "filet", "réseau"
    },

    "prison": {
        # Lieux
        "bagne", "taulard", "détention", "cellule", "cachot", "mitard", "quartier", "cour", "parloir", "promenade",
        "barreau", "grille", "porte", "serrure", "clé", "menotte", "chaîne", "cuffs", "combinaison", "uniforme",
        # Acteurs
        "détenu", "prisonnier", "condamné", "crim", "keuf", "maton", "gardien", "surveillant", "directeur", "avocat",
        "juge", "tribunal", "procès", "peine", "condamnation", "amnistie", "libération", "parole", "évasion",
        # Vie carcérale
        "règlement", "loi", "interdit", "punition", "isolement", "mitard", "bagarre", "violence", "gang", "clan",
        "drogue", "trafic", "deal", "dette", "protection", "respect", "hiérarchie", "pouvoir", "contrôle", "survie",
        # Émotions
        "colère", "rage", "haine", "désespoir", "tristesse", "solitude", "ennui", "peine", "souffrance", "regret",
        # Symboles
        "murs", "barbelés", "mirador", "tour", "salle", "douche", "cantine", "travail", "atelier", "bibliothèque"
    },

    "mort": {
        # Concepts
        "décès", "trépas", "fin", "disparition", "perte", "deuil", "enterrement", "funérailles", "cimetière", "tombe",
        # Causes
        "meurtre", "assassinat", "homicide", "crime", "balles", "couteau", "accident", "maladie", "cancer", "sida",
        "overdose", "suicide", "désespoir", "souffrance", "douleur", "agonie", "martyre", "sacrifice", "guerre", "attentat",
        # Symboles
        "squelette", "crâne", "os", "cercueil", "linceul", "croix", "pierre tombale", "épitaphe", "bougie", "fleur",
        # Métaphores
        "nuit", "obscurité", "froid", "glace", "néant", "vide", "silence", "oubli", "effacement", "disparition",
        # Émotions
        "tristesse", "chagrin", "pleurs", "larmes", "regret", "manque", "nostalgie", "souvenir", "mémoire", "héritage",
        # Croyances
        "âme", "esprit", "paradis", "enfer", "jugement", "réincarnation", "au-delà", "ciel", "ange", "démon"
    },

    "fête": {
        # Lieux
        "soirée", "boîte", "club", "after", "teuf", "rave", "concert", "festival", "scène", "backstage",
        # Ambiance
        "musique", "son", "basse", "rythme", "beat", "drop", "mix", "DJ", "MC", "rappeur",
        "lumière", "laser", "strobe", "fumée", "confettis", "feu d’artifice", "décor", "ambiance", "vibe", "énergie",
        # Boissons
        "alcool", "bière", "vin", "champagne", "whisky", "vodka", "rhum", "tequila", "shot", "cocktail",
        # Drogues
        "weed", "shit", "joint", "ecstasy", "MD", "pilule", "coke", "poudre", "défonce", "planer",
        # Actions
        "danser", "sauter", "bouger", "kiffer", "délirer", "crier", "chanter", "raper", "performer", "improviser",
        # Public
        "foule", "public", "fans", "groupies", "VIP", "bouncer", "sécurité", "file", "entrée", "billet",
        # Conséquences
        "gueule de bois", "fatigue", "sommeil", "réveil", "regret", "black-out", "perte de contrôle", "délire", "folie", "excès"
    },

    "sport": {
        # Disciplines
        "foot", "ballon", "terrain", "stade", "match", "but", "victoire", "défaite", "équipe", "coéquipier",
        "basket", "panier", "dribble", "shoot", "NBA", "streetball", "playground", "maillot", "sneakers", "chaussures",
        "boxe", "ring", "gants", "KO", "uppercut", "crochet", "jab", "combat", "champion", "ceinture",
        "musculation", "salle", "haltère", "barre", "poids", "gainage", "abdos", "pectoral", "biceps", "triceps",
        "course", "running", "marathon", "100m", "sprint", "endurance", "vitesse", "podium", "médaille", "or",
        # Valeurs
        "effort", "sueur", "douleur", "sacrifice", "discipline", "rigueur", "persévérance", "détermination", "volonté", "ambition",
        "respect", "fair-play", "esprit d’équipe", "leadership", "stratégie", "tactique", "entraînement", "coach", "mentor", "rival",
        # Symboles
        "trophée", "coupe", "championnat", "ligue", "classement", "record", "performance", "exploit", "légende", "mythe"
    },

    "mode": {
        # Vêtements
        "veste", "blouson", "doudoune", "hoodie", "sweat", "t-shirt", "chemise", "polo", "débardeur", "marcel",
        "pantalon", "jeans", "jogging", "survêt", "cargo", "short", "legging", "chaussette", "boxer", "caleçon",
        # Chaussures
        "baskets", "sneakers", "air max", "jordan", "adidas", "nike", "puma", "reebok", "new balance", "timberland",
        # Accessoires
        "casquette", "bonnet", "chaîne", "collier", "bague", "montre", "lunettes", "ceinture", "sac", "bandana",
        # Marques
        "gucci", "louis vuitton", "chanel", "dior", "prada", "supreme", "off-white", "balenciaga", "yeezy", "nike",
        # Style
        "streetwear", "hype", "luxe", "vintage", "retro", "swag", "style", "look", "tenue", "outfit",
        # Couleurs
        "noir", "blanc", "rouge", "bleu", "vert", "jaune", "or", "argent", "doré", "camouflage"
    },

    "voitures": {
        # Modèles
        "clio", "206", "golf", "bmw", "mercedes", "audi", "porsche", "ferrari", "lamborghini", "bugatti",
        "tesla", "range rover", "4x4", "suv", "berline", "sportive", "cabriolet", "moto", "scooter", "harley",
        # Éléments
        "moteur", "vitesse", "accélération", "0 à 100", "cheval", "puissance", "boîte", "embrayage", "volant", "siège",
        "jante", "pneu", "frein", "suspension", "échappement", "pot", "phare", "feu", "clignotant", "klaxon",
        # Actions
        "conduire", "rouler", "drifter", "burnout", "accélérer", "freiner", "démarrer", "caler", "réparer", "tuner",
        # Contexte
        "route", "autoroute", "boulevard", "rue", "parking", "garage", "station-service", "essence", "plein", "vidange",
        # Symboles
        "clé", "contact", "tableau de bord", "compteur", "gps", "radio", "son", "basse", "vitres", "toit"
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# DDL
# ─────────────────────────────────────────────────────────────────────────────

DDL_TRACK = """
CREATE TABLE IF NOT EXISTS tracks_analysis (

    -- ── Clés ──────────────────────────────────────────────────────────────────
    track_id                INTEGER NOT NULL,
    artist_id               INTEGER NOT NULL,
    album_id                INTEGER,
    track_name              VARCHAR,
    artist_name             VARCHAR,
    album_name              VARCHAR,
    isrc                    VARCHAR,

    -- ── 2. Statistiques lexicales ─────────────────────────────────────────────
    word_count              INTEGER,
    unique_word_count       INTEGER,
    ttr                     DOUBLE,
    avg_sentence_length     DOUBLE,
    sentence_count          INTEGER,
    top10_words             VARCHAR,    -- JSON

    -- ── 3. Sémantique ─────────────────────────────────────────────────────────
    lda_dominant_topic      INTEGER,
    lda_topic_keywords      VARCHAR,    -- JSON
    tfidf_top_keywords      VARCHAR,    -- JSON
    embedding_norm          DOUBLE,

    -- ── 4. Stylométrie ────────────────────────────────────────────────────────
    pos_noun_ratio          DOUBLE,
    pos_verb_ratio          DOUBLE,
    pos_adj_ratio           DOUBLE,
    pos_adv_ratio           DOUBLE,
    pos_pron_ratio          DOUBLE,
    pronoun_i_ratio         DOUBLE,
    pronoun_we_ratio        DOUBLE,
    pronoun_you_ratio       DOUBLE,

    -- ── 5. Rimes & structure ──────────────────────────────────────────────────
    rhyme_density           DOUBLE,
    avg_syllables_line      DOUBLE,
    repetition_ratio        DOUBLE,
    chorus_detected         BOOLEAN,

    -- ── 6. Lisibilité ─────────────────────────────────────────────────────────
    flesch_reading_ease     DOUBLE,
    flesch_kincaid_grade    DOUBLE,
    smog_index              DOUBLE,
    avg_word_length         DOUBLE,

    -- ── 7. Analyse avancée ────────────────────────────────────────────────────
    semantic_density        DOUBLE,
    lexical_diversity       DOUBLE,
    hapax_ratio             DOUBLE,
    
    -- ── 8. Émotions & champs lexicaux ────────────────────────────────────────
    emotion_scores          VARCHAR,    -- JSON 
    dominant_emotions       VARCHAR,    -- JSON
    lexical_field_scores    VARCHAR,    -- JSON
    dominant_lexical_fields VARCHAR,    -- JSON

    analyzed_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (track_id, artist_id)
);
"""

DDL_ALBUM = """
CREATE TABLE IF NOT EXISTS albums_analysis (

    -- ── Clés ──────────────────────────────────────────────────────────────────
    album_id                INTEGER NOT NULL,
    artist_id               INTEGER NOT NULL,
    album_name              VARCHAR,
    artist_name             VARCHAR,
    release_year            INTEGER,
    track_count             INTEGER,

    -- ── 2. Stats lexicales ────────────────────────────────────────────────────
    total_word_count        INTEGER,
    avg_word_count          DOUBLE,
    avg_unique_word_count   DOUBLE,
    avg_ttr                 DOUBLE,
    album_ttr               DOUBLE,
    avg_sentence_length     DOUBLE,
    album_vocabulary_size   INTEGER,
    top20_words             VARCHAR,    -- JSON

    -- ── 3. Sémantique ─────────────────────────────────────────────────────────
    lda_dominant_topic      INTEGER,
    lda_topic_distribution  VARCHAR,    -- JSON {topic_id: prob}
    tfidf_top_keywords      VARCHAR,    -- JSON
    intra_album_similarity  DOUBLE,

    -- ── 4. Stylométrie ────────────────────────────────────────────────────────
    avg_pos_noun_ratio      DOUBLE,
    avg_pos_verb_ratio      DOUBLE,
    avg_pos_adj_ratio       DOUBLE,
    avg_pos_adv_ratio       DOUBLE,
    avg_pos_pron_ratio      DOUBLE,
    avg_pronoun_i_ratio     DOUBLE,
    avg_pronoun_we_ratio    DOUBLE,
    avg_pronoun_you_ratio   DOUBLE,

    -- ── 5. Rimes & structure ──────────────────────────────────────────────────
    avg_rhyme_density       DOUBLE,
    avg_syllables_line      DOUBLE,
    avg_repetition_ratio    DOUBLE,
    pct_with_chorus         DOUBLE,

    -- ── 6. Lisibilité ─────────────────────────────────────────────────────────
    avg_flesch_reading_ease DOUBLE,
    avg_flesch_kincaid_grade DOUBLE,
    avg_smog_index          DOUBLE,
    avg_word_length         DOUBLE,

    -- ── 7. Avancé ─────────────────────────────────────────────────────────────
    avg_semantic_density    DOUBLE,
    avg_lexical_diversity   DOUBLE,
    avg_hapax_ratio         DOUBLE,
    album_hapax_ratio       DOUBLE,
    
    -- ── 8. Émotions & champs lexicaux ────────────────────────────────────────
    avg_emotion_scores          VARCHAR,    -- JSON moyennes par émotion
    dominant_emotions           VARCHAR,    -- JSON émotions les plus présentes
    avg_lexical_field_scores    VARCHAR,    -- JSON moyennes par champ
    dominant_lexical_fields     VARCHAR,    -- JSON champs les plus présents

    analyzed_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (album_id, artist_id)
);
"""

DDL_ARTIST = """
CREATE TABLE IF NOT EXISTS artists_analysis (

    -- ── Clés ──────────────────────────────────────────────────────────────────
    artist_id               INTEGER PRIMARY KEY,
    artist_name             VARCHAR,
    album_count             INTEGER,
    track_count             INTEGER,

    -- ── 2. Stats lexicales ────────────────────────────────────────────────────
    total_word_count        INTEGER,
    avg_word_count          DOUBLE,
    career_vocabulary_size  INTEGER,
    career_ttr              DOUBLE,
    avg_ttr                 DOUBLE,
    avg_unique_word_count   DOUBLE,
    avg_sentence_length     DOUBLE,
    top30_words             VARCHAR,    -- JSON

    -- ── 3. Sémantique ─────────────────────────────────────────────────────────
    lda_topic_distribution  VARCHAR,    -- JSON répartition topics carrière
    tfidf_top_keywords      VARCHAR,    -- JSON mots-clés signatures
    inter_album_similarity  DOUBLE,
    career_embedding_centroid VARCHAR,  -- JSON vecteur centroïde PCA 10d

    -- ── 4. Stylométrie ────────────────────────────────────────────────────────
    avg_pos_noun_ratio      DOUBLE,
    avg_pos_verb_ratio      DOUBLE,
    avg_pos_adj_ratio       DOUBLE,
    avg_pos_adv_ratio       DOUBLE,
    avg_pos_pron_ratio      DOUBLE,
    avg_pronoun_i_ratio     DOUBLE,
    avg_pronoun_we_ratio    DOUBLE,
    avg_pronoun_you_ratio   DOUBLE,
    style_signature         VARCHAR,    -- JSON feature vector normalisé

    -- ── 5. Rimes & structure ──────────────────────────────────────────────────
    avg_rhyme_density       DOUBLE,
    std_rhyme_density       DOUBLE,
    avg_syllables_line      DOUBLE,
    avg_repetition_ratio    DOUBLE,
    pct_with_chorus         DOUBLE,

    -- ── 6. Lisibilité ─────────────────────────────────────────────────────────
    avg_flesch_reading_ease DOUBLE,
    avg_flesch_kincaid_grade DOUBLE,
    avg_smog_index          DOUBLE,
    avg_word_length         DOUBLE,

    -- ── 7. Avancé ─────────────────────────────────────────────────────────────
    avg_semantic_density    DOUBLE,
    avg_lexical_diversity   DOUBLE,
    avg_hapax_ratio         DOUBLE,
    career_hapax_ratio      DOUBLE,
    
    -- ── 8. Émotions & champs lexicaux ────────────────────────────────────────
    avg_emotion_scores          VARCHAR,    -- JSON moyennes par émotion
    dominant_emotions           VARCHAR,    -- JSON émotions les plus présentes
    avg_lexical_field_scores    VARCHAR,    -- JSON moyennes par champ
    dominant_lexical_fields     VARCHAR,    -- JSON champs les plus présents

    analyzed_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

ALL_DDL = [DDL_TRACK, DDL_ALBUM, DDL_ARTIST]
