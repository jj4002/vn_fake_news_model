-- WARNING: This schema is for context only and is not meant to be run.
-- Table order and constraints may not be valid for execution.

CREATE TABLE public.news_corpus (
  id bigint NOT NULL DEFAULT nextval('news_corpus_id_seq'::regclass),
  title text NOT NULL,
  content text NOT NULL,
  url text NOT NULL UNIQUE,
  source character varying NOT NULL,
  published_date timestamp without time zone NOT NULL,
  embedding USER-DEFINED,
  created_at timestamp without time zone DEFAULT now(),
  CONSTRAINT news_corpus_pkey PRIMARY KEY (id)
);
CREATE TABLE public.reports (
  id bigint NOT NULL DEFAULT nextval('reports_id_seq'::regclass),
  video_id character varying NOT NULL,
  reported_prediction character varying NOT NULL,
  reason text,
  status character varying DEFAULT 'pending'::character varying CHECK (status::text = ANY (ARRAY['pending'::character varying, 'reviewed'::character varying, 'resolved'::character varying, 'rejected'::character varying]::text[])),
  created_at timestamp without time zone DEFAULT now(),
  CONSTRAINT reports_pkey PRIMARY KEY (id),
  CONSTRAINT reports_video_id_fkey FOREIGN KEY (video_id) REFERENCES public.videos(video_id)
);
CREATE TABLE public.videos (
  video_id character varying NOT NULL,
  video_url text NOT NULL UNIQUE,
  caption text,
  ocr_text text,
  stt_text text,
  author_id character varying,
  prediction character varying NOT NULL CHECK (prediction::text = ANY (ARRAY['REAL'::character varying, 'FAKE'::character varying]::text[])),
  confidence double precision NOT NULL CHECK (confidence >= 0::double precision AND confidence <= 1::double precision),
  method character varying NOT NULL CHECK (method::text = ANY (ARRAY['base_model'::character varying, 'rag_enhanced'::character varying]::text[])),
  created_at timestamp without time zone DEFAULT now(),
  report_count integer DEFAULT 0,
  CONSTRAINT videos_pkey PRIMARY KEY (video_id)
);