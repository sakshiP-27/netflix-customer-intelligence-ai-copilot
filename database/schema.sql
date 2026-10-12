-- Netflix Customer Intelligence — normalized schema for the nine cleaned tables.
-- Parent tables first, then tables that reference them.

DROP TABLE IF EXISTS churn_labels CASCADE;
DROP TABLE IF EXISTS customer_feedback CASCADE;
DROP TABLE IF EXISTS support_tickets CASCADE;
DROP TABLE IF EXISTS payments CASCADE;
DROP TABLE IF EXISTS viewing_activity CASCADE;
DROP TABLE IF EXISTS subscriptions CASCADE;
DROP TABLE IF EXISTS content CASCADE;
DROP TABLE IF EXISTS subscription_plans CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

CREATE TABLE customers (
    customer_id         VARCHAR(20) PRIMARY KEY,
    first_name          VARCHAR(100) NOT NULL,
    last_name           VARCHAR(100) NOT NULL,
    age                 INTEGER,
    gender              VARCHAR(50),
    country             VARCHAR(100),
    state_or_region     VARCHAR(100),
    city                VARCHAR(100),
    registration_date   DATE,
    acquisition_channel VARCHAR(100),
    customer_segment    VARCHAR(50),
    preferred_language  VARCHAR(50)
);

CREATE TABLE subscription_plans (
    plan_id         VARCHAR(20) PRIMARY KEY,
    plan_name       VARCHAR(50) NOT NULL,
    monthly_price   NUMERIC(10, 2) NOT NULL,
    video_quality   VARCHAR(50),
    max_devices     INTEGER,
    advertisements  VARCHAR(10)
);

CREATE TABLE content (
    content_id          VARCHAR(20) PRIMARY KEY,
    title               VARCHAR(255) NOT NULL,
    content_type        VARCHAR(50),
    genre               VARCHAR(100),
    release_year        INTEGER,
    duration_minutes    INTEGER,
    content_language    VARCHAR(50),
    production_country  VARCHAR(100),
    maturity_rating     VARCHAR(20)
);

CREATE TABLE subscriptions (
    subscription_id           VARCHAR(20) PRIMARY KEY,
    customer_id               VARCHAR(20) NOT NULL REFERENCES customers (customer_id),
    plan_id                   VARCHAR(20) NOT NULL REFERENCES subscription_plans (plan_id),
    subscription_start_date   DATE NOT NULL,
    subscription_end_date     DATE,
    subscription_status       VARCHAR(50),
    auto_renew                BOOLEAN,
    cancellation_date         DATE,
    cancellation_reason       VARCHAR(255),
    monthly_price             NUMERIC(10, 2)
);

CREATE TABLE viewing_activity (
    viewing_id                VARCHAR(20) PRIMARY KEY,
    customer_id               VARCHAR(20) NOT NULL REFERENCES customers (customer_id),
    content_id                VARCHAR(20) NOT NULL REFERENCES content (content_id),
    viewing_date              DATE,
    watch_duration_minutes    INTEGER,
    completion_percentage     NUMERIC(6, 2),
    device_type               VARCHAR(50),
    login_location            VARCHAR(100),
    session_duration_minutes  INTEGER
);

CREATE TABLE payments (
    payment_id              VARCHAR(20) PRIMARY KEY,
    customer_id             VARCHAR(20) NOT NULL REFERENCES customers (customer_id),
    subscription_id         VARCHAR(20) REFERENCES subscriptions (subscription_id),
    payment_date            DATE,
    amount                  NUMERIC(10, 2),
    payment_method          VARCHAR(50),
    payment_status          VARCHAR(50),
    failed_payment_reason   VARCHAR(255)
);

CREATE TABLE support_tickets (
    ticket_id                     VARCHAR(20) PRIMARY KEY,
    customer_id                   VARCHAR(20) NOT NULL REFERENCES customers (customer_id),
    ticket_date                   DATE,
    issue_category                VARCHAR(100),
    issue_subcategory             VARCHAR(100),
    priority                      VARCHAR(20),
    ticket_status                 VARCHAR(50),
    resolution_time_hours         NUMERIC(10, 2),
    customer_satisfaction_score   NUMERIC(4, 2),
    ticket_description            TEXT,
    support_agent_id              VARCHAR(20)
);

CREATE TABLE customer_feedback (
    feedback_id       VARCHAR(20) PRIMARY KEY,
    customer_id       VARCHAR(20) NOT NULL REFERENCES customers (customer_id),
    feedback_date     DATE,
    rating            INTEGER,
    feedback_text     TEXT,
    sentiment_label   VARCHAR(20)
);

CREATE TABLE churn_labels (
    customer_id    VARCHAR(20) PRIMARY KEY REFERENCES customers (customer_id),
    churned        BOOLEAN NOT NULL,
    churn_date     DATE,
    churn_reason   VARCHAR(255)
);

CREATE INDEX idx_subscriptions_customer_id ON subscriptions (customer_id);
CREATE INDEX idx_subscriptions_plan_id ON subscriptions (plan_id);
CREATE INDEX idx_viewing_activity_customer_id ON viewing_activity (customer_id);
CREATE INDEX idx_viewing_activity_content_id ON viewing_activity (content_id);
CREATE INDEX idx_payments_customer_id ON payments (customer_id);
CREATE INDEX idx_payments_subscription_id ON payments (subscription_id);
CREATE INDEX idx_support_tickets_customer_id ON support_tickets (customer_id);
CREATE INDEX idx_customer_feedback_customer_id ON customer_feedback (customer_id);
