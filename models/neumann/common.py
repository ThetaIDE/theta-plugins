import os


from .message_passing import MessagePassingModule
from .neumann import NEUMANN
from .reasoning_graph import ReasoningGraphModule
from .soft_logic import SoftLogic
from src.app.api.packages import provider_export
FactsConverter = provider_export('nsfr', 'FactsConverter')
from src.app.api.packages import provider_export
build_infer_module = provider_export('nsfr', 'build_infer_module')
get_blender_lang = provider_export('nsfr', 'get_blender_lang')
get_lang = provider_export('nsfr', 'get_lang')
from src.app.api.packages import provider_export
ValuationModule = provider_export('nsfr', 'ValuationModule')
resolve_ruleset = provider_export('nsfr', 'resolve_ruleset')


def get_neumann_model(env_name: str, rules: str, device: str, train=True, explain=False):
    current_path = os.path.dirname(__file__)
    lark_path = os.path.join(current_path, 'lark/exp.lark')
    resource = resolve_ruleset(env_name, rules)
    lang_base_path = str(resource.directory.parent) + "/"

    lang, clauses, bk, atoms = get_lang(lark_path, lang_base_path, resource.name)

    val_fn_path = str(resource.valuation_path)
    val_module = ValuationModule(val_fn_path, lang, device)

    # FC = FactsConverter(lang=lang, valuation_module=val_module, atoms=atoms, bk=bk, device=device)
    FC = FactsConverter(lang=lang, valuation_module=val_module, device=device)
    prednames = []
    for clause in clauses:
        if clause.head.pred.name not in prednames:
            prednames.append(clause.head.pred.name)
    m = len(prednames)
    # m = 5
    # IM = build_infer_module(clauses, atoms, lang, m=m, infer_step=2, train=train, device=device)
    # Neuro-Symbolic Forward Reasoner
    soft_logic = SoftLogic()
    MPM = MessagePassingModule(soft_logic=soft_logic, device=device, T=2)
    RGM = ReasoningGraphModule(clauses=clauses, facts=atoms, terms=lang.consts, lang=lang, max_term_depth=1, device=device)
    neumann = NEUMANN(facts_converter=FC, message_passing_module=MPM, reasoning_graph_module=RGM, program_size=m, atoms=atoms, bk=bk, clauses=clauses, device=device, train=train, explain=explain)
    return neumann


def get_blender_neumann_model(env_name: str, rules: str, device: str, train=False, mode='normal', explain=False):
    current_path = os.path.dirname(__file__)
    lark_path = os.path.join(current_path, 'lark/exp.lark')
    resource = resolve_ruleset(env_name, rules)
    lang_base_path = str(resource.directory.parent) + "/"

    lang, clauses, bk, atoms = get_blender_lang(lark_path, lang_base_path, resource.name)

    val_fn_path = str(resource.valuation_path)
    val_module = ValuationModule(val_fn_path, lang, device)

    FC = FactsConverter(lang=lang, valuation_module=val_module, device=device)
    prednames = []
    for clause in clauses:
        if clause.head.pred.name not in prednames:
            prednames.append(clause.head.pred.name)
    # if train:
    #     m = len(prednames)
    # else:
    #     m = len(clauses)
    m = len(clauses)
    # m = 5
    # Neuro-Symbolic Forward Reasoner
    soft_logic = SoftLogic()
    MPM = MessagePassingModule(soft_logic=soft_logic, device=device, T=2)
    RGM = ReasoningGraphModule(clauses=clauses, facts=atoms, terms=lang.consts, lang=lang, max_term_depth=1, device=device)
    neumann = NEUMANN(facts_converter=FC, message_passing_module=MPM, reasoning_graph_module=RGM, program_size=m, atoms=atoms, bk=bk, clauses=clauses, train=train)
    return neumann
