# Community Guidelines

Look, we need to talk about how we treat each other here. Not because we're checking a box, but because this actually matters.

## Why This Exists

Most code of conduct documents are copy-pasted corporate templates that nobody reads. We're not doing that. This project exists because people collaborate, and collaboration only works when people feel safe asking questions, making mistakes, and contributing their ideas.

We've seen communities fall apart because of toxicity, gatekeeping, and ego. We've also seen amazing things happen when people treat each other with basic respect and assume good intentions. That's what we're aiming for here.

## The Foundation: Assume Good Intent

Most conflicts happen because of misunderstandings, not malice. Someone writes a terse code review comment—are they being rude, or just busy? Someone asks a question that seems "obvious"—are they lazy, or just learning?

**Default to assuming people mean well.** If something feels off, ask for clarification before assuming the worst.

Real example: Last month, a contributor got a review comment that just said "This won't work." They were pretty upset—felt dismissive. Turns out the reviewer was on mobile, trying to help quickly between meetings, and forgot to add context. Once we talked it through, they explained the issue properly, everyone understood, and we moved on. Could've been drama, wasn't.

That said, assuming good intent doesn't mean tolerating bad behavior indefinitely. If someone's pattern of behavior is consistently hostile, that's different.

## What We Actually Care About

### Technical Discussions

**Be direct, but not harsh.** "This approach has performance issues because X" is great. "Did you even think about performance?" is not.

We've had really frank technical debates here. Someone wanted to make all dependencies required instead of optional. Another person strongly disagreed. The discussion got heated, but it stayed focused on trade-offs, use cases, and evidence. Nobody made it personal. That's the goal.

**It's okay to disagree.** Actually, it's often valuable. Just disagree about ideas, not people.

**Code reviews can feel personal.** Someone spent hours on that PR. When you review, remember you're critiquing code, not the person. "This function could be clearer" hits different than "You wrote this confusingly."

Real scenario: We once had a PR where the contributor really wanted to add a feature a certain way. Multiple reviewers suggested a different approach. The contributor felt like we were ganging up on them. We stepped back, explained the concerns more clearly, acknowledged the value in their approach, and found a middle ground. Could've been handled better from the start if we'd been more aware of how it might land emotionally.

### Asking Questions

**There are no stupid questions.** Seriously. We all start somewhere.

If someone asks something that's documented, just point them to the docs. No need for "RTFM" energy. Maybe the docs aren't clear. Maybe they're new. Maybe they're having a bad day and missed it. Doesn't matter—just help.

We've had people apologize for "basic" questions. Don't do that. Someone once asked "what's a tensor?" in our Discord. You know what happened? Three people jumped in with helpful explanations, and one linked to a great tutorial. That's what we want here.

**That said, show some effort.** "How do I install this?" when the README has install instructions—maybe try that first. But "I followed the install guide and got error X"—perfect, we can help with that.

### Mistakes Happen

You're going to break things. We all have.

Last year, someone accidentally force-pushed to main and deleted like 30 commits. Panic ensued. We restored from backup, tightened permissions, and moved on. The person was mortified, but honestly, we've all done similar things. It's fine.

**Own your mistakes, fix them if you can, and learn from them.** That's it. We're not going to shame you.

If you realize you were wrong in a discussion, it's totally okay to say "oh yeah, you're right, my bad." Actually, that's refreshing and makes everyone more comfortable admitting when they're wrong too.

### Giving and Receiving Feedback

**Feedback should be actionable.** "This code is bad" doesn't help anyone. "This function is doing too many things—consider splitting X and Y into separate functions" is useful.

**Sometimes feedback is wrong.** If you get a review comment that doesn't make sense, it's okay to push back or ask for clarification. We've had cases where reviewers suggested changes that would've actually made things worse. Polite disagreement is fine.

**Feedback is not a personal attack.** This is hard to internalize. Your code is not you. When someone suggests changes, they're trying to make the project better, not tell you you're bad at coding.

One contributor got really defensive during code review and said something like "well I guess I just suck at this." The reviewer felt terrible—they were just trying to help. We talked it through. Turns out the contributor had some past experiences with toxic communities. We adjusted how we gave feedback to them (more context, more encouragement), and over time they got more comfortable. Now they're one of our regular contributors and actually enjoy code reviews.

### Inclusivity vs. Tokenism

**Everyone's welcome here.** Doesn't matter where you're from, what language you speak, how much experience you have, or anything else.

But we're not going to make a big deal about diversity in some performative way. We care about building good software with good people. That includes people from all backgrounds, naturally.

If English isn't your first language and your comments are hard to follow sometimes, that's fine. We'll figure it out. If you're in a timezone where you can't make synchronous calls, that's fine too. We're async by default.

**Don't make assumptions about people.** We've had contributors who seemed junior but had 20 years of experience in a different domain. We've had people who seemed senior but were self-taught and still learning. You can't tell from a GitHub profile.

### The Gray Areas

Some things aren't black and white. Here's how we think about them:

**Tone in text is hard.** What sounds friendly in your head might read as curt to someone else. If someone tells you your message came across harshly, don't dismiss it. Maybe adjust next time. But also, don't overthink every word you write—we're all adults here.

**Humor and sarcasm are risky.** Among people who know each other, fine. With someone new? Easy to misread. We've had jokes land badly. When that happens, just apologize and move on.

**Technical elitism.** Look, some of us have been doing this for decades. Some of us started last year. The person with 15 years of experience is not inherently more valuable to this project. Different perspectives help.

That said, experience does matter when making technical decisions. If someone points out a security issue or performance problem based on hard-won experience, listen to them. But "I've been coding since before you were born" is not an argument.

**When is it okay to be blunt?** Security issues, critical bugs, or when someone's about to do something that'll cause real problems—yeah, be direct. "Stop, this will break production" is appropriate. But even then, you can be urgent without being a jerk.

**What about heated moments?** Sometimes discussions get intense. That's okay. But if you notice you're getting angry, step away. Come back when you've cooled down. We've all typed a message in frustration and then realized it would've made things worse. Delete it and write a better one.

### Things We Don't Tolerate

Most of this should be obvious, but let's be explicit:

**Harassment.** Don't target people. Don't dig through someone's history to find stuff to attack them with. Don't make someone feel unsafe.

**Bigotry.** Racist, sexist, homophobic, transphobic, ableist, or other discriminatory behavior—nope. We don't care if you meant it as a joke.

**Doxxing or threats.** Obvious dealbreaker.

**Sustained disruption.** If you're trolling, spamming, or deliberately derailing conversations, you'll be asked to stop. If it continues, you'll be removed.

**Unwanted sexual content or advances.** This is a technical project. Keep it professional.

We've only had to enforce this once. Someone was consistently dismissive and condescending in code reviews—not just direct, but actually insulting people. Multiple contributors mentioned it privately. We talked to them, they doubled down, we removed them from the project. It sucks, but sometimes it's necessary.

## What Happens If There's a Problem

**First, try to resolve it directly.** A lot of conflicts are just miscommunications. If someone said something that bothered you, consider replying and clarifying. Often that's all it takes.

**If that doesn't work or feels unsafe, reach out to maintainers.** You can:
- Email us privately (check the README for maintainer contacts)
- DM a maintainer on Discord
- Open a private issue if needed

**We'll figure it out together.** Our response depends on the situation. Maybe it's a misunderstanding we can clear up. Maybe someone needs to apologize. Maybe someone needs to change their behavior. In rare cases, maybe someone needs to be removed.

**We won't publicize complaints** unless necessary. If you report something, we'll handle it as discreetly as possible.

**What if you're accused of something?** We'll talk to you directly. You'll have a chance to explain. We're not looking to kick people out—we want to solve problems.

## Enforcement (The Boring But Necessary Part)

If you violate these guidelines, here's roughly what happens:

1. **First time (minor issue)**: Probably just a conversation. "Hey, that came across badly, maybe rephrase?"
2. **Repeated or more serious**: Formal warning, possibly temporary ban from parts of the project
3. **Severe or continued after warnings**: Removal from the project

We're not interested in being punitive. The goal is to keep the community functional and welcoming.

## This Document Will Evolve

We might have gotten things wrong here. As the community grows and we encounter new situations, we'll update this. If you think something should change, open an issue or discussion.

We don't pretend to have all the answers. We're figuring this out as we go, like everything else.

## The Actually Important Part

At the end of the day, this is simple: **treat people how you'd want to be treated if you were in their shoes.**

New contributor nervous about their first PR? Remember when that was you. Someone asks a question you think is obvious? Remember when nothing was obvious. Someone makes a mistake? Remember your mistakes.

We're all here because we think this project is interesting and want to make it better. That's the common ground. Let's build on that.

If you read this far, you probably didn't need to. You're already thinking about how to be a good community member. That's what matters.

Now go build something cool, help someone out, or ask that question you've been sitting on. We're glad you're here.
