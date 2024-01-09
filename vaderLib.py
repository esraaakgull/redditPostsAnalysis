import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import word_tokenize
from tabulate import tabulate

# languages
languages = ["A# .NET", "A# (Axiom)", "A-0 System", "A+", "A++", "ABAP", "ABC", "ABC ALGOL", "ABLE", "ABSET", "ABSYS",
             "ACC", "Accent", "Ace DASL", "ACL2", "ACT-III", "Action!", "ActionScript", "Ada", "Adenine", "Agda",
             "Agilent VEE", "Agora", "AIMMS", "Alef", "ALF", "ALGOL 58", "ALGOL 60", "ALGOL 68", "ALGOL W", "Alice",
             "Alma-0", "AmbientTalk", "Amiga E", "AMOS", "AMPL", "APL",
             "App Inventor for Android's visual block language", "AppleScript", "Arc", "ARexx", "Argus", "AspectJ",
             "Assembly language", "ATS", "Ateji PX", "AutoHotkey", "Autocoder", "AutoIt", "AutoLISP / Visual LISP",
             "Averest", "AWK", "Axum", "B", "Babbage", "Bash", "BASIC", "bc", "BCPL", "BeanShell",
             "Batch (Windows/Dos)", "Bertrand", "BETA", "Bigwig", "Bistro", "BitC", "BLISS", "Blue", "Bon", "Boo",
             "Boomerang", "Bourne shell", "bash", "ksh", "BREW", "BPEL", "C", "C--", "C++", "C#", "C/AL",
             "Caché ObjectScript", "C Shell", "Caml", "Candle", "Cayenne", "CDuce", "Cecil", "Cel", "Cesil", "Ceylon",
             "CFEngine", "CFML", "Cg", "Ch", "Chapel", "CHAIN", "Charity", "Charm", "Chef", "CHILL", "CHIP-8",
             "chomski", "ChucK", "CICS", "Cilk", "CL", "Claire", "Clarion", "Clean", "Clipper", "CLIST", "Clojure",
             "CLU", "CMS-2", "COBOL", "Cobra", "CODE", "CoffeeScript", "Cola", "ColdC", "ColdFusion", "COMAL",
             "Combined Programming Language", "COMIT", "Common Intermediate Language", "Common Lisp", "COMPASS",
             "Component Pascal", "Constraint Handling Rules", "Converge", "Cool", "Coq", "Coral 66", "Corn",
             "CorVision", "COWSEL", "CPL", "csh", "CSP", "Csound", "CUDA", "Curl", "Curry", "Cyclone", "Cython", "D",
             "DASL", "DASL", "Dart", "DataFlex", "Datalog", "DATATRIEVE", "dBase", "dc", "DCL", "Deesel", "Delphi",
             "DinkC", "DIBOL", "Dog", "Draco", "DRAKON", "Dylan", "DYNAMO", "E", "E#", "Ease", "Easy PL/I",
             "Easy Programming Language", "EASYTRIEVE PLUS", "ECMAScript", "Edinburgh IMP", "EGL", "Eiffel", "ELAN",
             "Elixir", "Elm", "Emacs Lisp", "Emerald", "Epigram", "EPL", "Erlang", "es", "Escapade", "Escher", "ESPOL",
             "Esterel", "Etoys", "Euclid", "Euler", "Euphoria", "EusLisp Robot Programming Language", "CMS EXEC",
             "EXEC 2", "Executable UML", "F", "F#", "Factor", "Falcon", "Fancy", "Fantom", "FAUST", "Felix", "Ferite",
             "FFP", "Fjölnir", "FL", "Flavors", "Flex", "FLOW-MATIC", "FOCAL", "FOCUS", "FOIL", "FORMAC", "@Formula",
             "Forth", "Fortran", "Fortress", "FoxBase", "FoxPro", "FP", "FPr", "Franz Lisp", "Frege", "F-Script",
             "FSProg", "G", "Google Apps Script", "Game Maker Language", "GameMonkey Script", "GAMS", "GAP", "G-code",
             "Genie", "GDL", "Gibiane", "GJ", "GEORGE", "GLSL", "GNU E", "GM", "Go", "Go!", "GOAL", "Gödel", "Godiva",
             "GOM (Good Old Mad)", "Goo", "Gosu", "GOTRAN", "GPSS", "GraphTalk", "GRASS", "Groovy",
             "Hack (programming language)", "HAL/S", "Hamilton C shell", "Harbour", "Hartmann pipelines", "Haskell",
             "Haxe", "High Level Assembly", "HLSL", "Hop", "Hope", "Hugo", "Hume", "HyperTalk",
             "IBM Basic assembly language", "IBM HAScript", "IBM Informix-4GL", "IBM RPG", "ICI", "Icon", "Id", "IDL",
             "Idris", "IMP", "Inform", "Io", "Ioke", "IPL", "IPTSCRAE", "ISLISP", "ISPF", "ISWIM", "J", "J#", "J++",
             "JADE", "Jako", "JAL", "Janus", "JASS", "Java", "Javascript", "JavaScript", "JCL", "JEAN", "Join Java",
             "JOSS", "Joule",
             "JOVIAL", "Joy", "JScript", "JScript .NET", "JavaFX Script", "Julia", "Jython", "K", "Kaleidoscope",
             "Karel", "Karel++", "KEE", "Kixtart", "KIF", "Kojo", "Kotlin", "KRC", "KRL", "KUKA", "KRYPTON", "ksh", "L",
             "L# .NET", "LabVIEW", "Ladder", "Lagoona", "LANSA", "Lasso", "LaTeX", "Lava", "LC-3", "Leda", "Legoscript",
             "LIL", "LilyPond", "Limbo", "Limnor", "LINC", "Lingo", "Linoleum", "LIS", "LISA", "Lisaac", "Lisp",
             "Lite-C", "Lithe", "Little b", "Logo", "Logtalk", "LPC", "LSE", "LSL", "LiveCode", "LiveScript", "Lua",
             "Lucid", "Lustre", "LYaPAS", "Lynx", "M2001", "M4", "Machine code", "MAD", "MAD/I", "Magik", "Magma",
             "Maple", "MAPPER", "MARK-IV", "Mary", "MASM Microsoft Assembly x86", "Mathematica", "MATLAB",
             "Maxima", "Macsyma", "Max", "MaxScript", "Maya (MEL)", "MDL", "Mercury", "Mesa", "Metacard", "Metafont",
             "MetaL", "Microcode", "MicroScript", "MIIS", "MillScript", "MIMIC", "Mirah", "Miranda", "MIVA Script",
             "ML", "Moby", "Model 204", "Modelica", "Modula", "Modula-2", "Modula-3", "Mohol", "MOO", "Mortran",
             "Mouse", "MPD", "CIL", "MSL", "MUMPS", "NASM", "NATURAL", "Napier88", "Neko", "Nemerle", "nesC", "NESL",
             "Net.Data", "NetLogo", "NetRexx", "NewLISP", "NEWP", "Newspeak", "NewtonScript", "NGL", "Nial", "Nice",
             "Nickle", "Nim", "NPL", "Not eXactly C", "Not Quite C", "NSIS", "Nu", "NWScript", "NXT-G", "o:XML", "Oak",
             "Oberon", "Obix", "OBJ2", "Object Lisp", "ObjectLOGO", "Object REXX", "Object Pascal", "Objective-C",
             "Objective-J", "Obliq", "Obol", "OCaml", "occam", "occam-π", "Octave", "OmniMark", "Onyx", "Opa", "Opal",
             "OpenCL", "OpenEdge ABL", "OPL", "OPS5", "OptimJ", "Orc", "ORCA/Modula-2", "Oriel", "Orwell", "Oxygene",
             "Oz", "P#", "ParaSail (programming language)", "PARI/GP", "Pascal", "Pawn", "PCASTL", "PCF", "PEARL",
             "PeopleCode", "Perl", "PDL", "PHP", "Phrogram", "Pico", "Picolisp", "Pict", "Pike", "PIKT", "PILOT",
             "Pipelines", "Pizza", "PL-11", "PL/0", "PL/B", "PL/C", "PL/I", "PL/M", "PL/P", "PL/SQL", "PL360", "PLANC",
             "Plankalkül", "Planner", "PLEX", "PLEXIL", "Plus", "POP-11", "PostScript", "PortablE", "Powerhouse",
             "PowerBuilder", "PowerShell", "PPL", "Processing", "Processing.js", "Prograph", "PROIV", "Prolog",
             "PROMAL", "Promela", "PROSE modeling language", "PROTEL", "ProvideX", "Pro*C", "Pure", "Python",
             "Q (equational programming language)", "Q (programming language from Kx Systems)", "Qalb", "QtScript",
             "QuakeC", "QPL", "R", "R++", "Racket", "RAPID", "Rapira", "Ratfiv", "Ratfor", "rc", "REBOL", "Red",
             "Redcode", "REFAL", "Reia", "Revolution", "rex", "REXX", "Rlab", "RobotC", "ROOP", "RPG", "RPL", "RSL",
             "RTL/2", "Ruby", "RuneScript", "Rust", "S", "S2", "S3", "S-Lang", "S-PLUS", "SA-C", "SabreTalk", "SAIL",
             "SALSA", "SAM76", "SAS", "SASL", "Sather", "Sawzall", "SBL", "Scala", "Scheme", "Scilab", "Scratch",
             "Script.NET", "Sed", "Seed7", "Self", "SenseTalk", "SequenceL", "SETL", "Shift Script", "SIMPOL", "SIGNAL",
             "SiMPLE", "SIMSCRIPT", "Simula", "Simulink", "SISAL", "SLIP", "SMALL", "Smalltalk", "Small Basic", "SML",
             "Snap!", "SNOBOL", "SPITBOL", "Snowball", "SOL", "Span", "SPARK", "Speedcode", "SPIN", "SP/k", "SPS",
             "Squeak", "Squirrel", "SR", "S/SL", "Stackless Python", "Starlogo", "Strand", "Stata", "Stateflow",
             "Subtext", "SuperCollider", "SuperTalk", "Swift", "Swift (Apple programming language)",
             "Swift (parallel scripting language)", "SYMPL", "SyncCharts", "SystemVerilog", "T", "TACL", "TACPOL",
             "TADS", "TAL", "Tcl", "Tea", "TECO", "TELCOMP", "TeX", "TEX", "TIE", "Timber", "TMG", "Tom", "TOM",
             "Topspeed", "TPU", "Trac", "TTM", "T-SQL", "TTCN", "Turing", "TUTOR", "TXL", "TypeScript", "Turbo C++",
             "Ubercode", "UCSD Pascal", "Umple", "Unicon", "Uniface", "UNITY", "Unix shell", "UnrealScript", "Vala",
             "VBA", "VBScript", "Verilog", "VHDL", "Visual Basic", "Visual Basic .NET", "Visual DataFlex",
             "Visual DialogScript", "Visual Fortran", "Visual FoxPro", "Visual J++", "Visual J#", "Visual Objects",
             "Visual Prolog", "VSXu", "Vvvv", "WATFIV, WATFOR", "WebDNA", "WebQL", "Windows PowerShell", "Winbatch",
             "Wolfram", "Wyvern", "X++", "X#", "X10", "XBL", "XC", "XMOS architecture", "xHarbour", "XL", "Xojo",
             "XOTcl", "XPL", "XPL0", "XQuery", "XSB", "XSLT", "XPath", "Xtend", "Yorick", "YQL", "Z notation", "Zeno",
             "ZOPL", "ZPL"]

# sentences examples
sentences = ["Surely if you wanna make lots of $ then you should learn COBOL",
             "If you're interested in learning programming/seeing how it works/see if it's for you, Python's cool..",
             "Python is really bad",
             'I took two years of Java and wanted to kill myself',
             'I prefer Javascript over Python.',
             'If you wanna eventually learn something else, Java. If you want to do the same thing for all eternity, Python.',
             'As someone trying to learn c++ from python, I hate that I started with python.',
             'If someone ask me, just go with Python. Versatile enough for most application.',
             'C# but that is my fanboy/addict self talking',
             'I think you should start with python to learn how programs work, then maybe move to C/C++',
             'C# will make you love programming compared to others suggested here',
             'Python is such a versatile and easy-to-learn language. I love how it simplifies complex tasks.',
             "Java has been my go-to programming language for years. It's reliable and performs well for large projects.",
             "I find JavaScript to be a bit challenging at first, but once you get the hang of it, it's powerful for web development.",
             "C++ gives me great control over memory and performance, but it can be tricky to manage pointers.",
             "Ruby is a pleasure to work with. Its clean syntax makes coding fun and productive.",
             "I'm not a big fan of PHP. It feels outdated, and there are better alternatives for web development.",
             "Go is lightning-fast and perfect for building scalable systems. I'm impressed with its performance.",
             "Swift has been a game-changer for iOS development. It's modern and makes coding enjoyable.",
             "I had a bad experience with Perl. Its syntax is confusing, and debugging can be a nightmare.",
             "Rust has become my new favorite language. Its safety features and performance are top-notch."
             ]

# data
data = [['SENTENCE', "LANGUAGE", "SENTIMENT"], ['________', "________", "_________"], ]


# analyzing the sentence if it is positive, negative or neutral
def analyze_sentiment(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    if sentiment_dict['compound'] >= 0.05:
        return "Positive"
    elif sentiment_dict['compound'] <= - 0.05:
        return "Negative"
    else:
        return "Neutral"


for sentence in sentences:
    words = word_tokenize(sentence)
    for word in words:
        if word in languages:
            polarity = analyze_sentiment(sentence.lower())
            row = [sentence, word, polarity]
            data.append(row)

print(tabulate(data))
