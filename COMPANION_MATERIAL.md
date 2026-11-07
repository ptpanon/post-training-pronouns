# Companion material

These passages were in the appendix of an earlier version of the paper and were cut when the appendix was reduced to what the nine pages use, in their vocabulary. They are kept here in the wording and the vocabulary they had in that version, so that nothing the paper once reported is lost; only LaTeX comments and internal provenance notes, which the submitted PDF never printed, were removed. Two figure references in the paper point here: the stage reading with the embedding estimator, and the interpolation of the preference delta. In all, 16 passages, about 3,242 words.

| # | passage | words |
|---|---|---|
| 1 | The grammatical mood swap (instrument ii) | 84 |
| 2 | Judges are secondary, and the cheap lexicon ruler is wrong-signed (instrument iii) | 185 |
| 3 | Why a word-level dominance lexicon points the wrong way here | 310 |
| 4 | Instruments that mislead, in full | 166 |
| 5 | Reading survives alignment | 77 |
| 6 | Undoing the preference delta | 88 |
| 7 | Training stages read with the embedding estimator | 127 |
| 8 | Gradients: the direction we could build does not behave like the estimator | 362 |
| 9 | The exchange: what the retreat leaves standing, in full | 259 |
| 10 | Four threads for future work | 140 |
| 11 | Exploratory profile map and reading cascade | 349 |
| 12 | Closing permutation check | 76 |
| 13 | The assistant-role anchor cell, in full | 305 |
| 14 | Figures: mood-swap ruler, stage estimator, preference-delta interpolation | 261 |
| 15 | Breadth, in one sentence | 77 |
| 16 | Does a reader read the impersonal form as less dominant? | 376 |

The text is LaTeX, as it stood in the paper. `\vekalet{...}` marks a paragraph drafted by the assistant under the project's review protocol and is otherwise a no-op.

## 1 · The grammatical mood swap (instrument ii)

```latex
\paragraph*{(ii) Grammatical mood swap.}
Rule-based minimal pairs (imperative $\leftrightarrow$ request) with the word set
held fixed, scored by teacher-forced NLL. The swap moves NLL in the same
direction under a short and a long frame across sixteen models
(Spearman $\rho = +0.90$), so the effect is not a frame-length artefact
(Appendix Figure~\ref{fig:mood}). \textbf{These are a different sixteen from the register
panel}, overlapping it in twelve and adding Falcon-H1-7B \citep{falconh1_2025},
Granite-4.1-8B \citep{granite2025card}, Mistral-Small-24B and Zephyr-7B
\citep{tunstall2023zephyr}; the two counts are not one panel measured twice.
```

## 2 · Judges are secondary, and the cheap lexicon ruler is wrong-signed (instrument iii)

```latex
\paragraph*{(iii) Judges are secondary, and the cheap lexicon ruler is wrong-signed.}% ★ v78 IS 1: §2'den AYNEN indi (iki ic gönderme Ek icine cevrildi)
Text a blind cross-family judge calls \textsc{dominant} scores \emph{lower} on
the Warriner dominance lexicon \citep{warriner2013norms} than text it calls \textsc{submissive}
($-0.21$, $-5.2\sigma$ at the item level; $-0.15$, $-2.8\sigma$ on intact text
alone, negative in $9$ of $11$ strata). On the mood swap ((ii) above), where the ground truth is
grammatical rather than judged, the same lexicon is silent on $54/85$ cross-tier
pairs and, on $26/31$ of the pairs where it does move, ranks the polite request
as the \emph{more} dominant text. \textbf{We therefore do not use it as a cheap substitute}.
\textbf{We ran the counterfactual}: pooled over every
response the base$\rightarrow$aligned difference on that lexicon falls in $7$
models and rises in $9$ --- no direction at all --- where second-person
density falls in every one. \textbf{The failure is one of category, not of
noise}; sentence embeddings are blind to the same contrast, and the annotation
literature reaches the same caution from its own side
(below, \emph{Instruments that mislead, in full}).
```

## 3 · Why a word-level dominance lexicon points the wrong way here

```latex
\paragraph*{Why a word-level dominance lexicon points the wrong way here.}
\vekalet{The instrument a reasonable study would reach for first fails on this
axis. A standard word-level dominance lexicon places \emph{would}, the single most
common carrier of English request framing, at the $78.4$th percentile of
dominance, above \emph{must} at $41.4$. Eight of the ten words that carry
request framing in our stimuli are absent from it altogether. Its dominance and
valence columns correlate at $\rho=+0.688$, against a permutation centre of
$-0.000$ with $\mathrm{sd}=0.008$, or $89$ null standard deviations, so what
looks like a dominance reading is substantially a pleasantness reading. On our
panels it points opposite to a blind judge.
\textbf{The counterfactual is no longer a direction: we ran it.} Pooling the
lexicon over every response and taking the base$\rightarrow$aligned difference
model by model, the pooled score falls in $7$ of $16$ models and rises in
$9$---no direction at all---while second-person density falls in $16$ of $16$ on
the same generations. A study that reached for this ruler alone would have read
these corpora as showing alignment \emph{increase} expressed dominance in more
than half the panel.
\textbf{The failure is one of category, not of noise.} A word-level norm records
the control a reader feels \emph{reading} a word; writer-dominance is carried by
structure---who is named as subject, whether an act is addressed---not by lexical
choice, and pooling per-word norms answers the first question in place of the
second. The dominance--valence entanglement is in the released norms themselves,
not in our pooling \citep{warriner2013norms}, which is why the mood swap, whose
ground truth is grammatical, is the primary evidence here.
The scope of the finding is the lexicon \emph{as used here}: a per-word dominance
average pooled over a response. Word-level affective norms
\citep{warriner2013norms,mohammad2018vad} are not in question as norms; what
fails is the pooling, and it is the pooled ruler we decline to use.}
```

## 4 · Instruments that mislead, in full

```latex
\paragraph*{Instruments that mislead, in full.}

%
\vekalet{The instrument a reasonable study would reach for first fails on this
axis. A standard word-level dominance lexicon reads substantially as a
pleasantness scale (the correlation and its permutation centre are given with
the instrument measurements below), and
\textbf{we ran the counterfactual rather than asserting it}: pooled over every
response the base$\rightarrow$aligned difference falls in $7$ models and rises
in $9$ --- no direction at all --- on the same generations where second-person
density falls in every one. A study reaching for this ruler alone would have read
alignment as \emph{increasing} expressed dominance in most of the panel.
\textbf{The failure is one of category, not of noise}: a word-level norm records
what a reader feels \emph{reading} a word; writer-dominance is carried by
structure. The released norms \citep{warriner2013norms,mohammad2018vad} are not
in question, only the per-word average pooled over a response
(Appendix~\ref{app:kesim}).}

\vekalet{Sentence embeddings are blind to the same contrast
(Appendix~\ref{app:kesim}).}
```

## 5 · Reading survives alignment

```latex
\paragraph{Reading survives alignment.}
\vekalet{On a fixed text, an aligned model's reading of directive force tracks
its base model's: across sixteen pairs and two argumentative substrates the
context an aligned model is given shifts its reading, but not in a way that
singles out the addressee, and the per-model outcomes do not line up into a
common pattern. We report the reading experiments as a map rather than a claim,
and give the full per-model cascade in
Appendix~\ref{app:kesim}.}
```

## 6 · Undoing the preference delta

```latex
\paragraph{Undoing the preference delta.}
Interpolating $\theta(\alpha)=\theta_{\mathrm{SFT}}+\alpha(\theta_{\mathrm{DPO}}
-\theta_{\mathrm{SFT}})$ moves the mood statistic monotonically in all four
ladders ($\rho=-1.000$ each), \textbf{the direction is shared}, yet the
curve \textbf{returns toward the pre-trained base in one of the four ladders}
(OLMo-2-13B); at 32B the same lab, recipe and grid give a curve that moves
\emph{away} from base (Figure~\ref{fig:delta}). One return out of four does not
identify what breaks it: scale is the difference we can name between those two
ladders, and with four points we report the count and not a cause.
```

## 7 · Training stages read with the embedding estimator

```latex
\paragraph{Rungs.}
Read rung by rung with the embedding estimator, the dominance shift is
\emph{built} at the SFT$\rightarrow$DPO rung in OLMo-2-13B and Tülu-3-8B, while
OLMo-2-32B moves the \emph{opposite} way at base$\rightarrow$SFT ($-2.6$
null-sd) and never recovers it, and six successive RLVR steps leave the axis
flat. Others localise post-training changes to a rung in the same way
\citep{liu2026alignmenttax,karouzos2026diversity}, the loss applied to prompt
tokens is known to matter
\citep{huertaenochian2024promptloss,chatterjee2025ittloss}, and length bias in
preference optimisation is a confound we control for rather than assume away
\citep{lu2024sampo}. \textbf{On this embedding estimator the rung that carries the change differs
between ladders of one recipe, so there it is not fixed by the recipe name}, and the name of that outcome can
itself depend on the protocol it is read under.
```

## 8 · Gradients: the direction we could build does not behave like the estimator

```latex
\paragraph{Gradients: the direction we could build does not behave like the
axis.}
Steering along a read-off direction is by now a standard move
\citep{park2023linear,zou2023repe,todd2023functionvectors,panickssery2023caa,
arditi2024refusal,turner2023actadd}, and preference optimisation has been
described as such a perturbation
\citep{raina2025dsteer,azizi2025asc,das2026spinal,kalajdzievski2025lims}.
This is the instrument on which the low-dose null of \S\ref{sec:dial} was
originally stated, and there are two measured reasons we state that null on the
counters instead and keep the direction here. First, it does not follow the
counters' sign. $\Delta U$ is read from generated text, not from weights: $U$ is how much further apart, in
sentence-embedding space, a checkpoint's continuations under dominant and under submissive in-context examples
move as the examples are multiplied, and $\Delta U$ is the supervised checkpoint's $U$ minus the arm's. At
$7.4\times$ the dose the force-selected arm moves it the furthest of the four arms, to
$\Delta U = +0.014$, while its own second-person count falls
(Table~\ref{tab:dial_dial}). Second, it did not pass its own manipulation test.
Injecting the pre-specified weight-space dominance direction moves aligned models
further orthogonal to that axis than along it, and a behavioural anchor built on
the same direction could not be established. The word \emph{orthogonal} is worth
replacing with the number it stands for. Comparing the real preference step with
the person-selected arm in weight space---both taken from the same supervised
checkpoint, over all $291$ tensors and $8.0$ billion coordinates---gives
$\cos = -0.0027$: \textbf{the two directions are orthogonal to within
$0.003$}. The sign of that residue is stable rather than large---a permutation
null that shuffles one vector's coordinates puts $0$ of $200$ draws at or beyond
it, the value is $70\times$ the isotropic baseline once corrected for the
anisotropy factor measured on this substrate, the largest single tensor carries
$4\%$ of either norm, and dropping the embedding and output matrices leaves
$\cos = -0.0027$. With eight billion coordinates a residue of this size is
resolvable without being a relation between the directions, so we report the
magnitude and make no claim from the sign. Two mechanical reasons were
measured rather than assumed: $99.9995\%$ of the weight-space direction's norm
sits in \texttt{lm\_head}, and a KL budget on the last position does not bound
the read quantity.
```

## 9 · The exchange: what the retreat leaves standing, in full

```latex
\paragraph*{The exchange: what the retreat leaves standing, in full.}%
%
\vekalet{Reading and production need not covary, because directive exchange is
not symmetric: the felicitous response to a command is usually compliance, not a
counter-command, so a model may register the imperative sharply and answer with
deference. Our map is consistent with this: reading and production axes are
not distinguishable from chance at the sizes we have, $n=6$ to $16$
depending on the pairing.} \vekalet{Read against \S\ref{sec:deperson}, the exchange has one fewer party than it appears to: the aligned model still performs the moves of correcting, evaluating and conditioning help, but performs them with neither party named. What survives, survives as an act without a subject or an object (Appendix~\ref{app:kesim}). One movement carries two descriptions: \emph{less human-like} in the anthropomorphism literature \citep{cheng2025humt,devrio2025taxonomy} and \emph{more polite} in the politeness-theoretic one \citep{brownlevinson1987}. We measure both descriptions of the same retreat, so we name both rather than choosing between them.}
\vekalet{Read through politeness theory, the aligned assistant performs one
of the oldest face-saving moves in the descriptive literature: it dissociates
itself and its reader from the correction it is making \citep{brownlevinson1987}.
What we do not know is whether the move is learned because raters reward it or
because it is the cheapest way to satisfy them; those are different claims, and
this paper measures only the first. That a preference pipeline encodes whose
preferences, and with what side effects, is a standing problem rather than a new
one \citep{casper2023open,santurkar2023whose,perez2022discovering}.}
```

## 10 · Four threads for future work

```latex
\paragraph{Four threads for future work.}
\vekalet{Four threads run through what we measured and are not separated by it:
\textbf{directive force} (how strongly an act is put), \textbf{status
assertion} (what standing the speaker claims), \textbf{conversational control}
(who holds the floor and sets the next move), and \textbf{accommodation} (how
far a speaker moves toward the other). Our instruments cut across all four
rather than along any one of them, which is the most likely reason the map in
Appendix~\ref{app:kesim} shows loosely coupled threads instead of a single factor.
Separating them is the work we would do next.} Register-conditioned
behaviour axes \citep{liu2026style} and language-dependent reversals of
alignment interventions \citep{fukui2026backfire} both suggest the same
thing from other directions: what alignment changes is not one quantity.
Minimal-pair methodology of the kind we use for the mood swap has a long
precedent in acceptability work \citep{suijkerbuijk2025blimpnl}.
```

## 11 · Exploratory profile map and reading cascade

```latex
\paragraph{Exploratory profile map and reading cascade, in brief.}
\label{sec:map}% eski etiket korunur
\label{sec:reading}% eski etiket korunur (dis atiflar kirilmasin)
\textbf{The map is exploratory: it carries no bar and no outcome name.}
\textbf{The cascade's registration carries both}: a two-sided permutation bar
at $0.05$ for every model and channel, and outcome names fixed before the added
models were read; its one failed prediction is in
Appendix~\ref{app:sapma}. We report both descriptively and give them here in
one paragraph, with the full material in the released repository rather than at
their original length.
%
Sixteen models against twenty measured axes, with empty cells left empty: on
the complete block the first component ($41\%$ of variance) puts all six reading
axes on one side and the mood and hedge axes on the other, which we state as a
shape we observed and not as a claim we tested. Read with the correlation result
above, the picture is several loosely coupled threads rather than one dominance
factor with several readouts. \textbf{The strongest correlation there is a
maximum statistic and its floor is not zero}: shuffling each axis within itself
($K=400$, missingness preserved) gives a mean maximum $|\rho|$ of $0.921$, and
against each pair's own reference every surviving pair is within-block, with
\textbf{no reading$\leftrightarrow$production pair surviving}. Clustering the
complete block recovers the \emph{producer} rather than the base weights:
Tülu-3-8B, which descends from Llama-3.1-8B, sits with the other AI2 models. On
the reading side, identical text scored under a real and a matched
pseudo-neutral context separates in $6/16$ models on one substrate and $5/16$
on the other, and two asymmetries are printed rather than smoothed: the
procedural-command channel changes sign between substrates, and the
dominant-assertion channel is nearly silent on one and negative on the other,
which is why the shared outcome is a general one rather than anything
addressee-specific. \textbf{The full cascade, per model and per channel, is in
the repository.}
```

## 12 · Closing permutation check

```latex
\paragraph{Closing permutation check.}
\textbf{This one we did measure against
chance.} Permuting the producer labels while holding the tree fixed
($K=400$) gives a mean adjusted Rand index of $+0.016$ with a 95th percentile
of $+0.217$. The observed value at four clusters is $+0.913$, and no
permutation reached it. The agreement is therefore not a reading of the
dendrogram's shape, but it remains a \emph{description} of thirteen
models, not a tested claim about producers in general.


% ============================================================================
```

## 13 · The assistant-role anchor cell, in full

```latex
clustering's chance reference is the permutation check above. The anchor cell, an
OLMo-2 pair, is the behavioural anchor of Appendix~\ref{app:D5}: the supervised weights are stepped either way along the
dominance direction at a fixed KL budget, the read quantity is the likelihood gap between the imperative and request forms of a
fixed set of minimal pairs, and $\mathrm{frac2}$ is the share of sign-randomised placebo directions that move that gap at least as far.
It ran to its full pre-registered size: all $60$ placebo
draws, $60$ of $60$ valid, and neither step moves the gap beyond its placebo
($\mathrm{frac2}=0.917$ and $0.083$ for the two perturbation signs).
It ran only after two repairs, and both are worth stating because they
are the kind of cost that usually goes unmeasured. First, memory: the in-place
route needs three model copies on one card (\textbf{76.6\,GiB}), while streaming
the direction and the reference copy from host memory brings the peak to
\textbf{28.4\,GiB}. Second, and larger, time: the streamed route was setting and
restoring the perturbed weights \emph{once per prompt}, twelve times per KL
evaluation, although the weight state is identical for all twelve. Hoisting that
out of the prompt loop cut weight operations \textbf{12-fold} and measured
\textbf{4.3--4.6$\times$} end-to-end (1300--1470\,s per draw $\rightarrow$
303--321\,s), and the rewrite was verified bit-identical to the original on CPU
($|\Delta|=0$ across four dtype/streaming configurations, and the check was shown
to fire on a $0.1\%$ perturbation). The draw count was \textbf{never reduced}.
We note a scope point the pre-registration did not settle: the placebo
null for this statistic is not centred at zero (means $+0.51$ and $+0.68$,
$|\mathrm{mean}|/\mathrm{sd}=0.90$ and $1.07$). Read against the measured centre
rather than against zero, $\mathrm{frac2}$ becomes $0.267$ and $0.100$, larger
than the $0.05$ bar either way, so \textbf{the outcome is unchanged}. We report
both readings rather than pick one after the fact.
```

## 14 · Figures: mood-swap ruler, stage estimator, preference-delta interpolation

```latex
\begin{figure}[tb]\centering
\includegraphics[width=\linewidth]{fig/F2_mood_ruler.pdf}
\caption{Left: the grammatical mood swap moves teacher-forced NLL in the same direction under a short and a long frame across 16 models (Spearman rho = +0.90), so the effect is not an artefact of frame length. Right: on the same swap, where the ground truth is grammatical rather than judged, the Warriner dominance lexicon is silent on 54/85 cross-tier pairs and, on 26/31 of the pairs where it does move, ranks the polite request as the MORE dominant text: the cheap ruler does not merely lack power here, it points the wrong way.}\label{fig:mood}
\end{figure}
\begin{figure}[tb]\centering
\includegraphics[width=\linewidth]{fig/F3_rungs.pdf}
\caption{The SFT$\rightarrow$DPO rung read twice, under the bare prompt and under each model's own chat template. Solid bars clear the two-sided empirical bar, faded bars do not, and a triangle marks the 4 counters whose outcome changes name between protocols. Right: the pre-specified embedding estimator on the same rungs. The chat template is not a nuisance parameter.}\label{fig:rungs}
\end{figure}
\begin{figure}[tb]\centering
\includegraphics[width=\linewidth]{fig/F4_dpo_delta.pdf}
\caption{Interpolating the preference delta $\theta$($\alpha$)=$\theta$\_SFT+$\alpha$($\theta$\_DPO$-$$\theta$\_SFT) moves the mood statistic monotonically in all four ladders ($\rho = -1.000$ each), yet only OLMo-2-13B returns toward its pre-trained base (star) when the delta is undone. At 32B the same lab, recipe and grid give a curve that moves away from base: the direction is shared and the return is not, and with one ladder at each size we do not name what separates them. The weight-space differencing follows the task-arithmetic construction \citep{ilharco2023task}; the alignment-tax literature applies the same subtraction to recover capability \citep{lin2024alignmenttax,wortsman2022soups}, and decoding-time methods use the same difference without touching the weights \citep{liu2024proxy,mitchell2024eft}.}\label{fig:delta}
\end{figure}
```

## 15 · Breadth, in one sentence

```latex
\textbf{Breadth, in one sentence.} We ran the dominance estimator on
twenty-four base$\rightarrow$aligned reads in all, fifteen end-to-end pairs
and nine ladder rungs, and the shift is predominantly one-signed there.
\textbf{The outcome name is protocol-dependent on five of the six rows where
both protocols were run}, so that breadth is a range of readings and not a
steady one; those reads belong to a separate instrument study, not to this
paper's contribution.


% ============================================================================
% ============================================================================

% ============================================================================
```

## 16 · Does a reader read the impersonal form as less dominant?

```latex
\paragraph{Does a reader read the impersonal form as less dominant?}
\vekalet{The register result is a claim about form. Whether the substitution is
what a reader registers as less dominant is a separate question, and we put it
to a blind test rather than assume it. From the panel we drew $276$ impersonal
correction sentences and made two minimal pairs from each: one against the
addressee-directed form (\emph{your claim}) and one against a third-person
control (\emph{their claim}), each shown in both orders, $1{,}104$ presentations
in all. Two judges answered which sentence places its writer higher above its
reader, one open-weight and read from its token distribution, one through an
agent interface. \textbf{Both readers are aligned models, and one of them is a
checkpoint from the panel itself} (Qwen2.5-32B-Instruct): the reader whose
uptake we are measuring belongs to the population whose register we measured, so
this test asks whether an aligned model reads the substitution as less dominant,
not whether a human does. The fifty-item human anchor below is the only
non-model reading here, and it was labelled by the author. The bar and the minimum detectable effect, $0.07$, were fixed
before any answer was seen. The personal form was read as the more dominant one
in every arm, by $+0.038$ ($95\%$ CI $[-0.005,+0.082]$) and $+0.047$
($[+0.004,+0.091]$) on the two readings of the open-weight arm and by $+0.043$
($[+0.027,+0.062]$) on the other, and \textbf{none of the three clears $0.07$}.
Two of the three intervals exclude zero, which is why we state the bar and not
the interval: an interval away from zero is not a threshold.
\textbf{The reason is a ceiling in the control arm}, and it is the finding worth
reporting: the third-person form was itself read as more dominant than the
impersonal one at $0.790$ and $0.957$, far above the $0.5$ of indifference.
Putting \emph{any} person back makes the sentence read as more dominant, so this
design cannot separate the addressee from a third party. A fifty-item human
anchor labelled by the author points the same way and more strongly ($+0.383$,
$[+0.250,+0.483]$), which is descriptive: the labeller knew the design. We
report this as a null for the reader-uptake question, not as evidence that the
form substitution has no effect on how a sentence is read.}
```

