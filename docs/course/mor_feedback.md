Hi all,

Thanks for sharing your ideas.

I hope the following points will be helpful and answer your questions:

While the directions you proposed are applied, they are still research projects and should be viewed as such. Namely, you should still be able to phrase this as a research question/problem and check the literature. Try to search for papers that did related things, for example, rhythm-aware generation in English or papers/tools that consider acronyms in Hebrew. Many tasks in NLP are applied (e.g., translation and summarization), but there's still a lot of literature about them.

How well will big LLMs do on these tasks? I.e., if you'd simply prompt Claude/ChatGPT to infer the acronym meaning from context, will it be able to do so? You need real motivation for your project. If this prompting will work, the question might still be interesting (is there a good reason why we would not want to use LLM?), but you'd still need to make a case for why the problem you're working on is relevant.

I think the two directions you proposed do have potential; you just need to make things a bit more concrete.

Regarding computation resources — the nodes we have in the cluster for the course have NVIDIA Titan XP and RTX 2080. You could do training but only of small models or training one of the parameter-efficient methods we saw in class.

Best,
Mor
