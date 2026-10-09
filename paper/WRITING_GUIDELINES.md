# Tips for Writing NLP Papers

Vered Shwartz, Medium, 16 Aug 2023. Full text as provided by the user, kept here as a private reference. Author's copyright; the Medium page navigation and the sign-up prompts are omitted.

Over the years I've developed a certain standard for writing papers (and doing research in general) that I share verbally with my students. I recently realized I'm repeating myself — and worse than that, editing the same things over and over again in paper drafts. So I decided to document my paper writing tips. This blog post is first and foremost intended for my students, although others might find it useful too.

Some of the tips here are specific to NLP papers, although many of them are general and might be useful for other fields as well. Some tips are standard practices while others are my own personal preferences. If you're not my student, feel free to ignore those ;)

As this standard keeps changing (hopefully improving!) I plan to update this blog post.

**Important Note:** This document doesn't replace the submission instructions for the particular venue you're targeting. Always follow the official submission instructions and style files if there are any conflicts between them and this document.

## Content

**Don't forget the why.** The paper needs to be clear about 1) what are the research questions it attempts to answer; and 2) why they are important. Make sure the introduction answers the "why" questions before diving into the "how" questions.

**Go from abstract to concrete.** Leave the more technical details to the method, data, and experimental setup sections. Always start with the more abstract details before diving into the specifics. A standard way to achieve this is to have a paragraph at the top of a technical section describing the subcomponents at a high level while referencing the specific subsection describing each in detail. In some cases it's also helpful to provide a visual overview of the proposed method. See this example paragraph from Ravi et al (2023):

> The architecture of our method is shown in Figure 1. We use the same clustering method as in Cattan et al. (2021a) but revise the pairwise scorer. Our goal is to improve the model's ability to resolve coreferences between mention pairs that are not lexically or contextually similar, but where one mention could be inferred from the other using commonsense knowledge and reasoning. Thus, we develop a commonsense inference engine (Sec 3.1) and use it to enhance the pairwise scorer (Sec 3.2).

**Write the introduction, abstract, and conclusion last.** I think it's helpful to think of the introduction as the summary of the paper and the abstract as a summary of the introduction. As such, they each need to address the following aspects: 1️⃣ motivation of the work, 2️⃣ gap from prior work, 3️⃣ proposed work, 4️⃣ findings, and 5️⃣ concluding sentence (typically pertaining to future work, limitations, or implications of the findings). For this reason, it's helpful to leave the writing of this section to the end, once the "narrative" of the paper is clear after describing the results. Here is an example abstract from Shwartz et al. (2020), where I marked each aspect:

> [1️⃣ Natural language understanding involves reading between the lines with implicit background knowledge.] [2️⃣ Current systems either rely on pre-trained language models as the sole implicit source of world knowledge, or resort to external knowledge bases (KBs) to incorporate additional relevant knowledge.] [3️⃣ We propose an unsupervised framework based on self-talk as a novel alternative to multiple-choice commonsense tasks. Inspired by inquiry-based discovery learning (Bruner, 1961), our approach inquires language models with a number of information seeking questions such as "what is the definition of…" to discover additional background knowledge.] [4️⃣ Empirical results demonstrate that the self-talk procedure substantially improves the performance of zero-shot language model baselines on four out of six commonsense benchmarks, and competes with models that obtain knowledge from external KBs.] [5️⃣ While our approach improves performance on several benchmarks, the self-talk induced knowledge even when leading to correct answers is not always seen as helpful by human judges, raising interesting questions about the inner-workings of pre-trained language models for commonsense reasoning.]

Another way to think about it is that the abstract and the introduction are your chances to convince the reader to read the entire paper. With that said, don't use techniques such as building suspense to attract readers. We're not writing fiction :) Spell out the main findings early on.

**Less is more.** You might have worked on the project for over a year. You probably implemented half a dozen solutions before finding the one that worked. In the paper, it's important that you only describe what's relevant, excluding some (often most) of the work you've done. Although research tends to involve many failed attempts and detours before landing on a successful idea, academic papers are structured as if you've had the one successful idea from the start.

*[Figure: Getting from point A to point B might have included many detours along the way, but the paper should (typically) describe it as a straight line.]*

On a related note, the paper doesn't have to describe what you did chronologically, it's more important to describe it in an order that is easy to understand.

Some students think that they need to highlight many different contributions in the paper. I think it doesn't always work in their favour. Reviewers are not impressed with more work, they want to see high quality work. Don't dilute your high quality solution with mediocre intermediate steps. Sometimes it's better to leave some ideas out, or to separate them to another paper.

There is an important distinction I'd like to make here. It's legitimate to experiment with different ideas and decide to leave some of them behind along the way. It's not legitimate to pick and choose experiments to report and exclude unflattering results. For example, let's say you're writing a method paper. You tested it on a few datasets where it performed well. Now you tested it on another dataset and it performs less well than the state-of-the-art or one of your baselines. You should still report the results.

**The paper needs to be self-contained.** Remember, you've worked on the problem for a long time and gained some knowledge that is not common knowledge in the field. Try to write the paper such that your lab mates working on different NLP tasks can easily understand it without reading prior work.

## Language

**Keep it simple.** It is very frustrating when papers use overly complicated language. Aim to write papers that can be understood by anyone in NLP, regardless of their subarea.

**Don't try to sound smart.** There is no need to use "sophisticated" words like *utilize* instead of *use*. It doesn't make you sound smarter or more academic.

**Be concise.** One of the advantages of the page limit in conference (and some journal) papers is that it encourages conciseness. Don't repeat your point within the same section (repeating some important points in the abstract, introduction, and the body of the paper is ok). Don't go back and forth between topics. Don't use vacuous words like *distinct* (as in "we evaluated our method on 5 distinct datasets" — it's implied that the 5 datasets are not the same) and *novel* (as in "we're proposing a novel method" — of course you're not proposing an existing method).

**Be clear and specific.** I was going to write "don't write abstract or vague sentences", but I realize this is an abstract and vague recommendation. So instead, let me be more specific: add examples. I especially like papers that follow the same example throughout the paper. For example, in Coil and Shwartz (2023), we ran with the "crocodile chocolate" example from the title, through the abstract and introduction, all the way to the analysis. But different examples are also ok. Please include examples!

**Don't write overly long sentences.** This advice applies to any genre of writing. It's cognitively difficult to understand long sentences. Split them into multiple short sentences.

**Don't oversell.** In recent years, there is a tendency to oversell papers as if each and every one of them is transformative — while in practice there is such a constant stream of NLP papers that most papers will hardly be remembered a few years down the line. I always prefer to be honest about the contributions of the paper (I even wrote a "limitations" section before it was mandatory for \*CL conferences!). For example, here is the end of the introduction from Waterson and Shwartz (2018):

> We experiment with various evaluation settings, including settings that make lexical memorization impossible. In these settings, the integrated method performs better than the baselines. Even so, the performance is mediocre for all methods, suggesting that the task is difficult and warrants further investigation.

We could have just mentioned that we have a method that outperforms the baselines. But I think it's important to not deceive the reader that this is a working solution that could be implemented into downstream tasks.

I realize this tip is tricky because being honest about the contributions and limitations makes your paper look worse — on the surface — than a paper written by an enthusiastic snake oil salesperson. But I'd really like to believe that there is a non-negligible number of reviewers out there who read beyond the hype and will appreciate your integrity.

## Comparison to Prior Work

**Know the literature.** Motivate your work in the context of prior work and existing problems. I've heard a tip circling around that grad students should not read the literature because it would make them less creative. That is, excuse me, complete bullshit. Most of us are not creative geniuses, and if you don't read the literature, the most likely thing that would happen is that you get to the paper writing part, do a quick Google search for the related work section, and discover that your idea has already been published before.

At the same time, you don't need to read every NLP paper as soon as it is published on arXiv. Try to dive deeply into the topic you're currently working on and choose other influential or interesting papers in other subareas more selectively.

**Don't plagiarize.** I can't stress this enough. It's not ok to copy complete sentences from other papers. The best way to make sure you don't plagiarize is to read the related work and summarize it in your own words (emphasizing the points more pertinent to your paper) in a document, and then consult the document when you write the related work section. Summarizing a paper right after you read it would also help you describe it in your paper in the context of your own work. I've seen cases of vague descriptions of prior work, sometimes using completely different terminology, only to google these sentences and find they were "borrowed" from the abstract of said paper. Needless to say, don't cite papers without reading them.

**The related work section is not a shopping list.** Don't just list papers that are related to yours. Try to group them according to their themes, draw conclusions, and use them to emphasize the gap in the literature that your work aims to address. For example, Qin et al. (2020) describe the related work along 3 dimensions, each paragraph ending with the differences from the current paper.

**Be critical.** Don't assume that anything written in previous work is gold. Yes, even if these are peer reviewed papers. If you doubt some claim written in previous work, you may be right about it. Test it carefully and challenge it if you find it to be wrong.

## Style

Ok, now we're getting into the nit-picky zone!

**Don't over-capitalize.** It's not Large Language Models, it's large language models.

**Don't define basic terms.** E.g. there is definitely no need to define NLP as "natural language processing" in an NLP paper.

**Quotation marks in LaTeX are different.** Opening quotation is ` `` ` and closing quotation is `''`. If you copy from a Google doc or some other document, make sure to replace the quotes. (For single quotes use `` ` `` and `'`).

**Use booktabs for tables.** Judge for yourself which one looks better:

*[Figures: regular vs booktabs table, from "Better LaTeX Tables with Booktabs"]*

## Citations

**Use the ACL anthology.** Most papers published in NLP venues are available in the ACL anthology. The conference style files typically include an `anthology.bib` file that includes all the bibtex entries for these papers (downloadable from here), as well as a `custom.bib` file for other citations. When you want to cite a paper, check if it exists on the ACL anthology. If it does, click on "cite" to see the bibtex key. For example:

*[Figure: ACL Anthology "cite" dialog]*

If it's not in the ACL anthology, find the bibtex from the other venue (or Google Scholar / Semantic Scholar) and put it in the `custom.bib` file.

**Cite the correct version of the paper.** Google Scholar often defaults to the arXiv version, but you should check if the paper has since been accepted at another venue (journal or conference), and cite that version instead.

**Use the correct citation format.** `\citep` is intended for parenthetical citations, as in "We extracted short narratives from the Toronto Book corpus (Zhu et al., 2015)". `\citet` (or `\newcite` in the \*CL style files) is used for citations within the text, as in "Yu and Ettinger (2020) showed that LMs encode the words that appear in a given text, but capture little information regarding phrase meaning".
