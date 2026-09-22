"""Proposal drafting: template-based full-draft generation for project proposals."""

from grantscout.proposal.pipeline import ProposalPipeline
from grantscout.proposal.real_templates import register_real_templates
from grantscout.proposal.schemas import ProposalDocument, ProposalRequest
from grantscout.proposal.templates import get_template, list_templates

register_real_templates()

__all__ = [
    "ProposalDocument",
    "ProposalPipeline",
    "ProposalRequest",
    "get_template",
    "list_templates",
]
