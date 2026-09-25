import json
import random
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
SEED = 5
N_DOCS = 30
N_QUESTIONS = 100
N_UNANSWERABLE = 25
N_MULTIHOP = 15
N_TEST = 30
N_TEST_UNANSWERABLE = 7
LAST_YEAR = 2025

TYPES = [
    ("m", "музей", "музея", "музею", "музей", "музеем", "музее"),
    ("f", "библиотека", "библиотеки", "библиотеке", "библиотеку", "библиотекой", "библиотеке"),
    ("m", "институт", "института", "институту", "институт", "институтом", "институте"),
    ("m", "завод", "завода", "заводу", "завод", "заводом", "заводе"),
    ("m", "театр", "театра", "театру", "театр", "театром", "театре"),
    ("f", "обсерватория", "обсерватории", "обсерватории", "обсерваторию", "обсерваторией", "обсерватории"),
    ("f", "фабрика", "фабрики", "фабрике", "фабрику", "фабрикой", "фабрике"),
    ("m", "технопарк", "технопарка", "технопарку", "технопарк", "технопарком", "технопарке"),
    ("f", "клиника", "клиники", "клинике", "клинику", "клиникой", "клинике"),
    ("f", "лаборатория", "лаборатории", "лаборатории", "лабораторию", "лабораторией", "лаборатории"),
    ("m", "колледж", "колледжа", "колледжу", "колледж", "колледжем", "колледже"),
    ("n", "издательство", "издательства", "издательству", "издательство", "издательством", "издательстве"),
    ("m", "ботанический сад", "ботанического сада", "ботаническому саду", "ботанический сад",
     "ботаническим садом", "ботаническом саду"),
    ("f", "киностудия", "киностудии", "киностудии", "киностудию", "киностудией", "киностудии"),
    ("n", "конструкторское бюро", "конструкторского бюро", "конструкторскому бюро", "конструкторское бюро",
     "конструкторским бюро", "конструкторском бюро"),
]

NAMES = [
    "Северный маяк", "Янтарная долина", "Полярная звезда", "Зеленый луч", "Серебряный ручей",
    "Каменный цветок", "Белая цапля", "Синяя гора", "Лунный свет", "Старая мельница",
    "Золотой колос", "Ветер перемен", "Красный мост", "Тихая гавань", "Морской узел",
    "Ясный полдень", "Дальний берег", "Северное сияние", "Липовый парк", "Огненная птица",
    "Новая заря", "Горный хрусталь", "Звездный путь", "Весенний сад", "Южный крест",
    "Хрустальный родник", "Медный всадник", "Голубая лагуна", "Лесная сказка", "Белый парус",
]

CITIES = [
    "Верхнеозерск", "Светлоярск", "Белоборск", "Ольхогорск", "Кедроборск", "Туманоярск",
    "Ледогорск", "Малиноозерск", "Ручьевск", "Зарябинск", "Липоборск", "Камышеярск",
    "Рябинозерск", "Тихоборск", "Дубравинск", "Клюквинск", "Бережанск", "Волнорецк",
    "Соколоярск", "Черноборск", "Гранитогорск", "Мшанск",
]

STREETS = [
    "Липовая", "Садовая", "Речная", "Заводская", "Звездная", "Кленовая", "Северная", "Озерная",
    "Каменная", "Солнечная", "Мостовая", "Луговая", "Березовая", "Полевая", "Школьная", "Лесная",
]

MALE = ["Андрей", "Борис", "Виктор", "Григорий", "Дмитрий", "Евгений", "Игорь", "Константин",
        "Леонид", "Михаил", "Николай", "Олег", "Павел", "Роман", "Сергей", "Тимофей", "Федор", "Юрий"]
FEMALE = ["Анна", "Вера", "Галина", "Дарья", "Елена", "Жанна", "Ирина", "Ксения", "Лариса",
          "Марина", "Надежда", "Ольга", "Полина", "Светлана", "Татьяна", "Юлия"]
SURNAMES = ["Громов", "Белов", "Карпов", "Лебедев", "Морозов", "Никитин", "Орлов", "Панин",
            "Родионов", "Соколов", "Тарасов", "Уваров", "Федотов", "Хромов", "Чернов", "Шилов",
            "Яковлев", "Воронин", "Гаврилов", "Демидов", "Ершов", "Жуков", "Зимин", "Ильин",
            "Калинин", "Ломов", "Мельников", "Нестеров", "Осипов", "Прохоров", "Рябов", "Суханов",
            "Титов", "Ушаков", "Филатов", "Щукин", "Абрамов", "Буров", "Власов", "Горбунов"]

BRANCHES = ["Северный", "Южный", "Восточный", "Западный", "Приморский", "Речной", "Горный", "Лесной"]
AWARDS = ["Золотой циркуль", "Хрустальный компас", "Серебряный якорь", "Бронзовая сова", "Янтарное перо",
          "Малахитовый ключ", "Стальной маятник", "Изумрудный лист", "Жемчужная раковина", "Медный колокол",
          "Гранитная ступень", "Лазурный парус"]
PARTNERS = [("компанией", "Альфа-Вектор"), ("фондом", "Открытый горизонт"), ("университетом", "Политех-Север"),
            ("агентством", "Мост-Инфо"), ("компанией", "Кварц-Систем"), ("фондом", "Наследие края"),
            ("компанией", "Орбита-Плюс"), ("агентством", "Север-Медиа"), ("университетом", "Техно-Юг"),
            ("компанией", "Ладья-Софт")]

FILLERS = [
    "{T_nom} регулярно проводит открытые лекции и экскурсии для жителей {city_gen}.",
    "Сотрудники {T_gen} участвуют в городских праздниках и ярмарках.",
    "Работа {T_gen} освещается в местной прессе и на сайте городской администрации.",
    "Руководство уделяет большое внимание обучению молодых специалистов.",
    "В архиве хранятся фотографии, письма и документы разных лет.",
    "Каждую осень сюда приходят студенты на практику.",
    "За годы работы {T_nom} стал(а|о) заметной частью жизни {city_gen}.",
    "Многие сотрудники работают здесь больше двадцати лет.",
    "Для посетителей оборудованы гардероб, кафе и небольшая комната отдыха.",
    "Годовые отчеты публикуются в открытом доступе.",
    "В здании есть конференц-зал на сто мест.",
    "Местные школы часто приводят сюда учеников на занятия.",
    "Ветераны коллектива ежегодно собираются на встречу в начале декабря.",
    "При {T_prep} действует совет попечителей из представителей бизнеса и науки.",
    "Внутренние правила пересматриваются раз в несколько лет.",
    "Летом часть сотрудников уходит в отпуск, но основная работа не прекращается.",
    "Газеты {city_gen} не раз писали о необычных проектах коллектива.",
    "Руководство поддерживает связи с коллегами из других регионов.",
    "Своим главным богатством здесь считают людей, а не здания и оборудование.",
    "На прилегающей территории высажены липы и клены.",
    "Официальный сайт {T_gen} обновляется каждую неделю.",
    "В выходные дни часть помещений открыта для экскурсий.",
    "Отдел кадров проводит собеседования с кандидатами круглый год.",
    "О прошлом {T_gen} рассказывает небольшая экспозиция в холле.",
    "Некоторые решения руководства вызывали споры в городском совете.",
    "Бухгалтерия ежегодно проходит внешнюю проверку.",
    "В последние годы документы активно переводятся в электронный вид.",
    "Сотрудники выпускают внутренний бюллетень о новостях и планах.",
    "Попасть на открытые мероприятия можно по предварительной записи.",
    "Директор ежегодно отчитывается о работе перед советом попечителей.",
    "Филиалы регулярно обмениваются опытом и проводят совместные семинары.",
    "Смена руководства каждый раз сопровождалась обновлением планов развития.",
    "Штатное расписание утверждается в начале каждого года.",
    "Бюджет формируется из городских средств, грантов и собственных доходов.",
    "Здание не раз ремонтировали, но фасад сохранил исторический облик.",
    "Коллектив гордится наградами, полученными за годы работы.",
    "Партнерские проекты помогают привлекать дополнительное финансирование.",
    "Первые годы работы были трудными: не хватало помещений и оборудования.",
    "В военные годы работа почти остановилась, но архив удалось сохранить.",
    "Со временем выросло число направлений работы и сотрудников.",
    "Многие выпускники местного университета начинали карьеру именно здесь.",
    "В коридорах висят портреты людей, которые руководили коллективом в разные годы.",
    "По традиции новый сезон открывается торжественным собранием коллектива.",
    "Здесь бережно относятся к истории и собирают воспоминания старых сотрудников.",
    "Городские власти не раз называли {T_acc} примером для других организаций.",
    "На сайте можно найти расписание, новости и контакты отделов.",
    "Посетители отмечают доброжелательность персонала и удобное расположение.",
    "Рядом со зданием находится остановка общественного транспорта.",
    "Отчеты о расходах проверяет независимая комиссия.",
    "Новые сотрудники проходят обязательный вводный курс.",
    "Часть оборудования была закуплена по программе обновления материальной базы.",
    "Старое здание до сих пор используется как склад и мастерская.",
    "В юбилейные годы выпускаются памятные буклеты и фотоальбомы.",
    "Представители коллектива выступают на региональных конференциях.",
    "Здесь действует программа наставничества для молодых специалистов.",
]


def g(gender, m, f, n):
    return {"m": m, "f": f, "n": n}[gender]


def cap(s):
    return s[:1].upper() + s[1:]


def street_forms(adj):
    return {"nom": adj, "gen": adj[:-2] + "ой", "acc": adj[:-2] + "ую", "prep": adj[:-2] + "ой"}


def adj_ins(adj):
    return adj[:-2] + ("им" if adj.endswith("ий") else "ым")


def workers(n):
    if 11 <= n % 100 <= 14 or n % 10 > 4 or n % 10 == 0:
        return f"{n} сотрудников"
    return f"{n} сотрудника"


def people(n):
    if 11 <= n % 100 <= 14 or n % 10 not in (2, 3, 4):
        return f"{n} человек"
    return f"{n} человека"


def city_forms(city):
    return {"nom": city, "gen": city + "а", "prep": city + "е"}


class Person:
    def __init__(self, first, surname, female):
        self.female = female
        self.surname = surname + "а" if female else surname
        self.name = f"{first} {self.surname}"
        self.key = surname.lower()

    def v(self, male, female):
        return female if self.female else male


class Doc:
    def __init__(self, idx, rnd, org_type, name, city):
        self.rnd = rnd
        self.id = f"doc_{idx:02d}"
        gender, *forms = org_type
        self.gender = gender
        self.T = dict(zip(["nom", "gen", "dat", "acc", "ins", "prep"], forms))
        self.name = name
        self.city = city_forms(city)
        self.paragraphs = []
        self.deck = []
        self.questions = {"single": [], "multi": [], "unans": []}
        self.used_surnames = set()
        self.other_cities = [c for c in CITIES if c != city]

    def ctx(self):
        n = f"«{self.name}»"
        return {f"T_{k}": f"{v} {n}" for k, v in self.T.items()} | {
            "city_gen": self.city["gen"], "city_prep": self.city["prep"], "city_nom": self.city["nom"]}

    def fill(self, template):
        text = template.format(**self.ctx())
        text = text.replace("стал(а|о)", g(self.gender, "стал", "стала", "стало"))
        return cap(text)

    def gv(self, m, f, n):
        return g(self.gender, m, f, n)

    def person(self):
        female = self.rnd.random() < 0.45
        surname = self.rnd.choice([s for s in SURNAMES if s not in self.used_surnames])
        self.used_surnames.add(surname)
        return Person(self.rnd.choice(FEMALE if female else MALE), surname, female)

    def filler(self):
        if not self.deck:
            self.deck = self.rnd.sample(FILLERS, len(FILLERS))
        return self.fill(self.deck.pop())

    def add(self, text, kind, extra=True):
        if extra and self.rnd.random() < 0.6:
            text = text + " " + self.filler()
        self.paragraphs.append({"text": text, "kind": kind})
        return len(self.paragraphs) - 1

    def q(self, cat, question, answer, key, evidence, qtype):
        self.questions[cat].append({"question": question, "answer": answer, "answer_key": key,
                                    "evidence": evidence, "type": qtype})


def years_between(rnd, lo, hi, k, exclude=()):
    pool = [y for y in range(lo, hi + 1) if y not in exclude]
    return sorted(rnd.sample(pool, k))


def build_doc(idx, rnd, org_type, name, city):
    d = Doc(idx, rnd, org_type, name, city)
    c = d.ctx()
    T, N = d.T, f"«{d.name}»"
    founded = rnd.randint(1890, 1965)
    founder = d.person()

    d.add(f"{cap(T['nom'])} {N}: справочная информация", "title", extra=False)
    p_found = d.add(rnd.choice([
        f"{cap(c['T_nom'])} {d.gv('был основан', 'была основана', 'было основано')} в {founded} году в {c['city_prep']}. "
        f"{d.gv('Его', 'Ее', 'Его')} {founder.v('основателем', 'основательницей')} считается {founder.name}.",
        f"История {c['T_gen']} началась в {founded} году, когда {founder.name} {founder.v('открыл', 'открыла')} "
        f"{T['acc']} в {c['city_prep']}.",
    ]), "founding")
    d.q("single", rnd.choice([f"В каком году {d.gv('был основан', 'была основана', 'было основано')} {c['T_nom']}?",
                              f"Когда {d.gv('появился', 'появилась', 'появилось')} {c['T_nom']}?"]),
        str(founded), str(founded), [p_found], "год основания")
    d.q("single", rnd.choice([f"Кто основал {c['T_acc']}?",
                              f"Кто считается основателем {c['T_gen']}?"]),
        founder.name, founder.key, [p_found], "основатель")

    start = founded + rnd.randint(8, 35)
    n_dir = rnd.choice([3, 4])
    bounds = years_between(rnd, start + 4, 2021, n_dir - 1)
    while any(b2 - b1 < 4 for b1, b2 in zip([start] + bounds, bounds + [LAST_YEAR])):
        bounds = years_between(rnd, start + 4, 2021, n_dir - 1)
    periods = list(zip([start] + bounds, bounds + [None]))
    directors = []
    for a, b in periods:
        p = d.person()
        if b is None:
            text = rnd.choice([
                f"С {a} года {c['T_ins']} руководит {p.name}. "
                f"До назначения {p.v('он', 'она')} много лет {p.v('работал', 'работала')} в других организациях {c['city_gen']}.",
                f"Нынешний директор {c['T_gen']} {p.name} {p.v('занял', 'заняла')} эту должность в {a} году.",
            ])
        else:
            text = rnd.choice([
                f"С {a} по {b} год {c['T_ins']} {p.v('руководил', 'руководила')} {p.name}.",
                f"В {a} году директором {c['T_gen']} {p.v('был назначен', 'была назначена')} {p.name}; "
                f"{p.v('он оставался', 'она оставалась')} на этом посту до {b} года.",
                f"{p.name} {p.v('возглавлял', 'возглавляла')} {T['acc']} с {a} по {b} год.",
            ])
        pid = d.add(text, "director")
        directors.append((a, b or LAST_YEAR + 1, p, pid))
    boundaries = {a for a, _, _, _ in directors} | {b for _, b, _, _ in directors}

    def director_at(year):
        for a, b, p, pid in directors:
            if a < year < b:
                return p, pid
        return None, None

    for a, b, p, pid in rnd.sample(directors, 2):
        inner = [y for y in range(a + 1, min(b, LAST_YEAR)) if y not in boundaries]
        if inner:
            y = rnd.choice(inner)
            d.q("single", rnd.choice([f"Кто руководил {c['T_ins']} в {y} году?",
                                      f"Кто занимал пост директора {c['T_gen']} в {y} году?"]),
                p.name, p.key, [pid], "руководитель в году")
    if start - founded >= 3:
        y = rnd.randint(founded + 1, start - 1)
        d.q("unans", rnd.choice([f"Кто руководил {c['T_ins']} в {y} году?",
                                 f"Кто был директором {c['T_gen']} в {y} году?"]),
            None, None, [], "руководитель до известной хронологии")

    emp_years = years_between(rnd, start, 2024, 3)
    for y in emp_years:
        n = rnd.randint(40, 1900)
        pid = d.add(rnd.choice([
            f"В {y} году штат {c['T_gen']} насчитывал {workers(n)}.",
            f"По данным на {y} год, в {c['T_prep']} работали {people(n)}.",
            f"К {y} году численность персонала {c['T_gen']} составила {people(n)}.",
        ]), "employees")
        d.q("single", rnd.choice([f"Сколько сотрудников работало в {c['T_prep']} в {y} году?",
                                  f"Какой была численность персонала {c['T_gen']} в {y} году?",
                                  f"Сколько человек числилось в штате {c['T_gen']} в {y} году?"]),
            str(n), str(n), [pid], "сотрудники в году")
    y = rnd.choice([y for y in range(start, 2025) if y not in emp_years])
    d.q("unans", rnd.choice([f"Сколько сотрудников работало в {c['T_prep']} в {y} году?",
                             f"Какой была численность персонала {c['T_gen']} в {y} году?"]),
        None, None, [], "сотрудники в отсутствующем году")

    bud_years = years_between(rnd, 2005, 2024, 2)
    for y in bud_years:
        m = rnd.randint(12, 950)
        pid = d.add(rnd.choice([
            f"Годовой бюджет {c['T_gen']} в {y} году составил {m} млн рублей.",
            f"В {y} году на содержание {c['T_gen']} было выделено {m} млн рублей.",
        ]), "budget")
        d.q("single", rnd.choice([f"Каким был годовой бюджет {c['T_gen']} в {y} году?",
                                  f"Сколько денег было потрачено на {c['T_acc']} в {y} году?"]),
            f"{m} млн рублей", str(m), [pid], "бюджет в году")
    y = rnd.choice([y for y in range(2005, 2025) if y not in bud_years])
    d.q("unans", f"Каким был годовой бюджет {c['T_gen']} в {y} году?", None, None, [], "бюджет в отсутствующем году")

    street_now, street_old = [street_forms(s) for s in rnd.sample(STREETS, 2)]
    house = rnd.randint(2, 88)
    p_addr = d.add(rnd.choice([
        f"Сейчас {c['T_nom']} располагается на улице {street_now['prep']}, дом {house}.",
        f"Адрес {c['T_gen']}: {c['city_nom']}, улица {street_now['nom']}, {house}.",
    ]), "address")
    d.q("single", rnd.choice([f"На какой улице сейчас находится {c['T_nom']}?",
                              f"Какой адрес у {c['T_gen']}?"]),
        f"улица {street_now['nom']}, {house}", street_now["nom"][:-2].lower(), [p_addr], "адрес")

    event_years = []
    if rnd.random() < 0.8:
        y_move = rnd.choice([y for y in range(start + 1, 2024) if y not in boundaries])
        event_years.append(y_move)
        p_move = d.add(rnd.choice([
            f"В {y_move} году {c['T_nom']} {d.gv('переехал', 'переехала', 'переехало')} с улицы {street_old['gen']} "
            f"на улицу {street_now['acc']}.",
            f"До {y_move} года {c['T_nom']} {d.gv('занимал', 'занимала', 'занимало')} старое здание на улице "
            f"{street_old['prep']}, после чего {d.gv('переехал', 'переехала', 'переехало')} в новое.",
        ]), "move")
        d.q("single", rnd.choice([f"На какой улице {d.gv('находился', 'находилась', 'находилось')} {c['T_nom']} до переезда?",
                                  f"Где {d.gv('располагался', 'располагалась', 'располагалось')} {c['T_nom']} раньше, до смены здания?"]),
            f"улица {street_old['nom']}", street_old["nom"][:-2].lower(), [p_move], "старый адрес")
        p, pid = director_at(y_move)
        if p:
            d.q("multi", rnd.choice([f"Кто руководил {c['T_ins']} в год переезда в новое здание?",
                                     f"Кто был директором {c['T_gen']}, когда {d.gv('он', 'она', 'оно')} "
                                     f"{d.gv('переехал', 'переехала', 'переехало')} на улицу {street_now['acc']}?"]),
                p.name, p.key, [p_move, pid], "руководитель в год события")
    if rnd.random() < 0.7:
        y_rec = rnd.choice([y for y in range(start + 1, 2024) if y not in boundaries and y not in event_years])
        event_years.append(y_rec)
        cost = rnd.randint(15, 700)
        p_rec = d.add(rnd.choice([
            f"В {y_rec} году здание {c['T_gen']} закрыли на реконструкцию; работы обошлись в {cost} млн рублей.",
            f"Капитальный ремонт {c['T_gen']} начался в {y_rec} году, его стоимость составила {cost} млн рублей.",
        ]), "reconstruction")
        d.q("single", rnd.choice([f"Сколько стоила реконструкция здания {c['T_gen']}?",
                                  f"Во что обошлось обновление здания {c['T_gen']}?"]),
            f"{cost} млн рублей", str(cost), [p_rec], "стоимость реконструкции")
        d.q("single", f"В каком году началась реконструкция {c['T_gen']}?", str(y_rec), str(y_rec), [p_rec], "год реконструкции")
        p, pid = director_at(y_rec)
        if p:
            d.q("multi", rnd.choice([f"Кто руководил {c['T_ins']}, когда началась реконструкция здания?",
                                     f"При каком директоре {c['T_acc']} закрыли на ремонт?"]),
                p.name, p.key, [p_rec, pid], "руководитель в год события")
    else:
        d.q("unans", rnd.choice([f"Сколько стоила реконструкция здания {c['T_gen']}?",
                                 f"В каком году здание {c['T_gen']} закрывали на капитальный ремонт?"]),
            None, None, [], "отсутствующее событие")

    award_names = rnd.sample(AWARDS, rnd.choice([1, 2]))
    award_years = []
    for award in award_names:
        y = rnd.choice([y for y in range(start + 1, 2025) if y not in boundaries and y not in award_years])
        award_years.append(y)
        pid = d.add(rnd.choice([
            f"В {y} году {c['T_nom']} {d.gv('получил', 'получила', 'получило')} премию «{award}».",
            f"Премия «{award}» была присуждена {c['T_dat']} в {y} году.",
        ]), "award")
        d.q("single", rnd.choice([f"Какую премию {d.gv('получил', 'получила', 'получило')} {c['T_nom']} в {y} году?",
                                  f"Какой наградой {d.gv('был отмечен', 'была отмечена', 'было отмечено')} {c['T_nom']} в {y} году?"]),
            f"«{award}»", award.lower(), [pid], "премия в году")
        d.q("single", f"В каком году {c['T_nom']} {d.gv('получил', 'получила', 'получило')} премию «{award}»?",
            str(y), str(y), [pid], "год премии")
        p, did = director_at(y)
        if p:
            d.q("multi", f"Кто руководил {c['T_ins']}, когда {d.gv('он', 'она', 'оно')} "
                         f"{d.gv('получил', 'получила', 'получило')} премию «{award}»?",
                p.name, p.key, [pid, did], "руководитель в год события")
    y = rnd.choice([y for y in range(start + 1, 2025) if y not in award_years])
    d.q("unans", rnd.choice([f"Какую премию {d.gv('получил', 'получила', 'получило')} {c['T_nom']} в {y} году?",
                             f"Какой наградой {d.gv('был отмечен', 'была отмечена', 'было отмечено')} {c['T_nom']} в {y} году?"]),
        None, None, [], "премия в отсутствующем году")

    n_br = rnd.choice([2, 2, 3])
    br_names = rnd.sample(BRANCHES, n_br + 1)
    br_cities = rnd.sample(d.other_cities, n_br + 1)
    for bname, bcity in zip(br_names[:n_br], br_cities[:n_br]):
        bc = city_forms(bcity)
        y = rnd.randint(start + 1, 2023)
        head = d.person()
        p_open = d.add(rnd.choice([
            f"{bname} филиал {c['T_gen']} открылся в {y} году в {bc['prep']}.",
            f"В {y} году {c['T_nom']} {d.gv('открыл', 'открыла', 'открыло')} {bname.lower()} филиал в {bc['prep']}.",
            f"В {y} году в {bc['prep']} начал работу {bname.lower()} филиал {c['T_gen']}.",
        ]), "branch")
        p_head = d.add(rnd.choice([
            f"{bname} филиал сейчас возглавляет {head.name}.",
            f"{head.name} руководит {adj_ins(bname.lower())} филиалом {c['T_gen']}.",
        ]), "branch_head")
        d.q("single", rnd.choice([f"В каком году открылся {bname.lower()} филиал {c['T_gen']}?",
                                  f"Когда начал работу {bname.lower()} филиал {c['T_gen']}?"]),
            str(y), str(y), [p_open], "год открытия филиала")
        d.q("single", f"В каком городе находится {bname.lower()} филиал {c['T_gen']}?",
            bcity, bcity.lower(), [p_open], "город филиала")
        d.q("single", f"Кто возглавляет {bname.lower()} филиал {c['T_gen']}?", head.name, head.key, [p_head], "глава филиала")
        d.q("multi", rnd.choice([f"В каком городе работает филиал {c['T_gen']}, которым руководит {head.name}?",
                                 f"Где расположен филиал {c['T_gen']}, который возглавляет {head.name}?"]),
            bcity, bcity.lower(), [p_open, p_head], "город филиала по руководителю")
    missing_city = city_forms(br_cities[-1])
    d.q("unans", rnd.choice([f"В каком году открылся филиал {c['T_gen']} в {missing_city['prep']}?",
                             f"Кто возглавляет филиал {c['T_gen']} в {missing_city['prep']}?"]),
        None, None, [], "несуществующий филиал")
    d.q("unans", f"Кто возглавляет {br_names[-1].lower()} филиал {c['T_gen']}?", None, None, [], "несуществующий филиал")

    ptype, pname = rnd.choice(PARTNERS)
    y = rnd.randint(2000, 2024)
    p_part = d.add(f"В {y} году {c['T_nom']} {d.gv('подписал', 'подписала', 'подписало')} соглашение о сотрудничестве "
                   f"с {ptype} «{pname}».", "partner")
    d.q("single", rnd.choice([f"С какой организацией {c['T_nom']} {d.gv('подписал', 'подписала', 'подписало')} "
                              f"соглашение о сотрудничестве?",
                              f"Кто стал партнером {c['T_gen']} в {y} году?"]),
        f"с {ptype} «{pname}»", pname.lower(), [p_part], "партнер")

    if rnd.random() < 0.5:
        for y in years_between(rnd, 2015, 2024, 2):
            n = rnd.randint(3, 250) * 100
            pid = d.add(f"В {y} году мероприятия {c['T_gen']} посетили {people(n)}.", "visitors")
            d.q("single", rnd.choice([f"Сколько человек посетили мероприятия {c['T_gen']} в {y} году?",
                                      f"Какой была посещаемость мероприятий {c['T_gen']} в {y} году?"]),
                str(n), str(n), [pid], "посетители в году")
    else:
        d.q("unans", rnd.choice([f"Сколько человек посетили мероприятия {c['T_gen']} в 2019 году?",
                                 f"Какой была посещаемость мероприятий {c['T_gen']} в 2022 году?"]),
            None, None, [], "отсутствующий показатель")

    n_facts = len(d.paragraphs)
    target = rnd.randint(max(20, n_facts + 3), 50)
    seen = {p["text"] for p in d.paragraphs}
    while len(d.paragraphs) < target:
        text = " ".join(d.filler() for _ in range(rnd.choice([2, 2, 3])))
        if text not in seen:
            seen.add(text)
            d.paragraphs.append({"text": text, "kind": "filler"})
    return d


def shuffle_body(rnd, d):
    order_rank = {"title": 0, "founding": 1, "director": 2, "move": 3, "reconstruction": 3, "address": 4,
                  "employees": 5, "budget": 5, "branch": 6, "branch_head": 6, "award": 7, "partner": 7,
                  "visitors": 8, "filler": None}
    idx = list(range(len(d.paragraphs)))
    facts = sorted([i for i in idx if d.paragraphs[i]["kind"] != "filler"],
                   key=lambda i: (order_rank[d.paragraphs[i]["kind"]], i))
    fillers = [i for i in idx if d.paragraphs[i]["kind"] == "filler"]
    order = facts[:1]
    rest = facts[1:]
    slots = sorted(rnd.sample(range(len(rest) + len(fillers)), len(fillers)))
    it_f, it_r = iter(fillers), iter(rest)
    for pos in range(len(rest) + len(fillers)):
        order.append(next(it_f) if pos in slots else next(it_r))
    new_id = {old: new + 1 for new, old in enumerate(order)}
    d.paragraphs = [d.paragraphs[i] for i in order]
    for qs in d.questions.values():
        for q in qs:
            q["evidence"] = sorted(new_id[i] for i in q["evidence"])


def main():
    rnd = random.Random(SEED)
    DATA.mkdir(exist_ok=True)
    (DATA / "docs").mkdir(exist_ok=True)
    types = (TYPES * 3)[:N_DOCS]
    rnd.shuffle(types)
    names = rnd.sample(NAMES, N_DOCS)
    docs = []
    for i in range(N_DOCS):
        city = rnd.choice(CITIES)
        d = build_doc(i + 1, rnd, types[i], names[i], city)
        shuffle_body(rnd, d)
        docs.append(d)
        (DATA / "docs" / f"{d.id}.txt").write_text("\n\n".join(p["text"] for p in d.paragraphs) + "\n", encoding="utf-8")

    per_doc = [4] * (N_QUESTIONS - 3 * N_DOCS) + [3] * (N_DOCS - (N_QUESTIONS - 3 * N_DOCS))
    rnd.shuffle(per_doc)
    cats = ["unans"] * N_UNANSWERABLE + ["multi"] * N_MULTIHOP + ["single"] * (N_QUESTIONS - N_UNANSWERABLE - N_MULTIHOP)
    rnd.shuffle(cats)
    slots = [(d, k) for d, k in zip(docs, per_doc)]
    questions, pos = [], 0
    for d, k in slots:
        used_q = set()
        for _ in range(k):
            cat = cats[pos]
            pos += 1
            pool = [q for q in d.questions[cat] if q["question"] not in used_q] or \
                   [q for q in d.questions["single"] if q["question"] not in used_q]
            q = rnd.choice(pool)
            used_q.add(q["question"])
            questions.append({"doc_id": d.id, **q, "answerable": q["answer"] is not None})
    rnd.shuffle(questions)

    unans = [q for q in questions if not q["answerable"]]
    ans = [q for q in questions if q["answerable"]]
    test = rnd.sample(unans, N_TEST_UNANSWERABLE) + rnd.sample(ans, N_TEST - N_TEST_UNANSWERABLE)
    test_ids = {id(q) for q in test}
    for i, q in enumerate(questions, 1):
        q["id"] = f"q{i:03d}"
        q["split"] = "test" if id(q) in test_ids else "train"

    def dump(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    gold_fields = ["id", "doc_id", "question", "answer", "answer_key", "evidence", "answerable", "type"]
    dump(DATA / "questions_train.jsonl",
         [{**{k: q[k] for k in gold_fields if k != "evidence"}, "evidence_ids": q["evidence"]}
          for q in questions if q["split"] == "train"])
    dump(DATA / "questions_test.jsonl",
         [{"id": q["id"], "doc_id": q["doc_id"], "question": q["question"]} for q in questions if q["split"] == "test"])
    dump(DATA / "test_gold.jsonl",
         [{**{k: q[k] for k in gold_fields if k not in ("evidence", "question")}, "evidence_ids": q["evidence"]}
          for q in questions if q["split"] == "test"])
    dump(DATA / "docs_meta.jsonl", [{"doc_id": d.id, "type": d.T["nom"], "name": d.name, "city": d.city["nom"],
                                     "n_paragraphs": len(d.paragraphs)} for d in docs])

    sizes = [len(d.paragraphs) for d in docs]
    print(f"docs: {len(docs)}, paragraphs: {sum(sizes)} (min {min(sizes)}, max {max(sizes)})")
    print(f"questions: {len(questions)}, unanswerable: {len(unans)}, multi-hop: "
          f"{sum(len(q['evidence']) > 1 for q in questions)}, test: {len(test)}")


if __name__ == "__main__":
    main()
