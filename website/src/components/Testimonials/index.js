import React from 'react';

const testimonials = [
  {
    quote: "Neural DSL cut our prototyping time by 60%. The ability to switch between TensorFlow and PyTorch seamlessly is a game-changer for our research team.",
    author: "Dr. Sarah Chen",
    title: "AI Research Lead, Stanford Medical AI Lab",
    avatar: "SC"
  },
  {
    quote: "The built-in debugger saved us weeks of debugging time. We found and fixed gradient vanishing issues in our 50-layer network within hours.",
    author: "Alex Rodriguez",
    title: "ML Engineer, TechCorp",
    avatar: "AR"
  },
  {
    quote: "As a beginner, I found Neural DSL incredibly easy to learn. The documentation and examples are excellent. I built my first working model in under an hour.",
    author: "Emily Johnson",
    title: "Data Science Student, MIT",
    avatar: "EJ"
  },
  {
    quote: "Shape validation alone is worth it. No more runtime shape errors! The automatic shape propagation catches issues before I even run the code.",
    author: "Michael Zhang",
    title: "Senior ML Engineer, AutoDrive Labs",
    avatar: "MZ"
  },
  {
    quote: "We deployed Neural DSL across our entire ML team. The standardization on one DSL improved collaboration and reduced onboarding time significantly.",
    author: "Lisa Wang",
    title: "VP of Engineering, ShopMart",
    avatar: "LW"
  },
  {
    quote: "The HPO integration is fantastic. Running cross-framework hyperparameter optimization from a single config file simplified our workflow immensely.",
    author: "David Park",
    title: "Research Scientist, VoiceAI",
    avatar: "DP"
  },
];

function Testimonial({quote, author, title, avatar}) {
  return (
    <div className="testimonial">
      <p className="testimonial__quote">"{quote}"</p>
      <div className="testimonial__author">
        <div 
          className="testimonial__avatar"
          style={{
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'white',
            fontWeight: 'bold'
          }}>
          {avatar}
        </div>
        <div>
          <div className="testimonial__name">{author}</div>
          <div className="testimonial__title">{title}</div>
        </div>
      </div>
    </div>
  );
}

export default function Testimonials() {
  return (
    <section className="testimonials">
      <div className="container">
        <h2 className="text--center margin-bottom--lg">
          Trusted by Developers Worldwide
        </h2>
        <div className="testimonial-grid">
          {testimonials.map((props, idx) => (
            <Testimonial key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
