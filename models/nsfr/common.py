import os

from src.app.api.rulesets import resolve_ruleset
from src.app.api.workspace import WorkspacePaths

from .facts_converter import FactsConverter
from .nsfr import NSFReasoner
from .utils.logic import build_infer_module, get_blender_lang, get_lang
from .valuation import ValuationModule


def get_nsfr_model(env_name: str, rules: str, device: str, train=False, explain=False):
    current_path = os.path.dirname(__file__)
    lark_path = os.path.join(current_path, 'lark/exp.lark')
    resource = resolve_ruleset(WorkspacePaths.for_process().root, env_name, rules)
    lang_base_path = str(resource.directory.parent) + "/"

    lang, clauses, bk, atoms = get_lang(lark_path, lang_base_path, resource.name)

    val_fn_path = str(resource.valuation_path)
    val_module = ValuationModule(val_fn_path, lang, device)

    FC = FactsConverter(lang=lang, valuation_module=val_module, device=device)
    prednames = []
    for clause in clauses:
        if clause.head.pred.name not in prednames:
            prednames.append(clause.head.pred.name)
    m = len(prednames) if train else len(clauses)
    # m = 5
    IM = build_infer_module(clauses, atoms, lang, m=m, infer_step=2, train=train, device=device)
    # Neuro-Symbolic Forward Reasoner
    NSFR = NSFReasoner(facts_converter=FC, infer_module=IM, atoms=atoms, bk=bk, clauses=clauses, device=device, train=train, explain=explain)
    return NSFR


def get_blender_nsfr_model(env_name: str, rules: str, device: str, train=False, mode='normal', explain=False):
    current_path = os.path.dirname(__file__)
    lark_path = os.path.join(current_path, 'lark/exp.lark')
    resource = resolve_ruleset(WorkspacePaths.for_process().root, env_name, rules)
    lang_base_path = str(resource.directory.parent) + "/"

    lang, clauses, bk, atoms = get_blender_lang(lark_path, lang_base_path, resource.name)

    val_fn_path = str(resource.valuation_path)
    val_module = ValuationModule(val_fn_path, lang, device)
    if hasattr(val_module, "module") and hasattr(val_module.module, "set_valuation_mode"):
        if str(rules).lower() in ["continuous", "soft"]:
            val_module.module.set_valuation_mode("continuous")
        elif str(rules).lower() in ["rigid", "default"]:
            val_module.module.set_valuation_mode("rigid")

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
    IM = build_infer_module(clauses, atoms, lang, m=m, infer_step=2, train=train, device=device)
    # Neuro-Symbolic Forward Reasoner
    NSFR = NSFReasoner(facts_converter=FC, infer_module=IM, atoms=atoms, bk=bk, clauses=clauses, device=device, train=train, explain=explain)
    return NSFR
