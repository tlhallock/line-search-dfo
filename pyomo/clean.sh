#!/bin/bash
find -name '*.pyomo.nl' -delete
find -name '*.pyomo.sol' -delete
find images -name '*.png' -delete
find images -name '*_debug_output.txt' -delete
find images -name 'different_ellipses_*' -delete
rm -f images/evaluations.txt
find images -name 'log.txt' -delete


















Safety/Success
    Errors: API

    Page load latency
        we have only implemented cold page load latency

    MSA Spectrum Wifi Install:
        definition: visit__application_details__application_name = 'MySpectrum' AND message__name = 'selectAction' AND state__view__current_page__page_name = 'accountHome' AND state__view__current_page__elements__standardized_name = 'wiFiProfile'

    Workflow completion rate: SMB Support Articles:
        definition: visit__application_details__application_name = 'SMB' AND message__name = 'selectAction' AND state__view__current_page__page_section__name = 'primaryArticles' AND state__view__current_page__elements__standardized_name = 'selectArticle'

    Workflow completion rate: One-time payment
        MSA

    Workflow completion rate: Equipment troubleshooting

    Not implemented:
        IVA resolution rate
        Live Contact Rate

More example behavioral metrics:
    Button clicks: SMB: "Yes" helpful article
        definition: visit__application_details__application_name = 'SMB' AND message__name = 'selectAction' AND state__view__current_page__page_section__name = 'wasThisArticleHelpful' AND state__view__current_page__elements__standardized_name = 'articleHelpful'

    Workflow completion rate: Auto-pay
        definition: visit__application_details__application_name = 'MySpectrum' AND message__name = 'pageView' AND state__view__current_page__page_name = 'paySuccessAutoPay'

    Button Clicks: SMB: "Search"'
        visit__application_details__application_name = 'SMB' AND message__name = 'selectAction' AND state__view__current_page__elements__standardized_name IN ('searchExpand', 'Open-search-box')

    Resi: Payment completion: Card/EFT
        'pay-bill.onetime.payment-method.checking' EFT
        'pay-bill.onetime.payment-method.credit' Credit
        'pay-bill.onetime.payment-method.savings' EFT

    Page views: SMB: Support
        definition: visit__application_details__application_name = 'SMB' AND message__name = 'pageViews' AND state__view__current_page__app_section = 'support'

    Button clicks: Continue Troubleshooting
        definition: SUM(IF(LOWER(visit_application_detailsapplication_name) = LOWER('SpecNet') AND LOWER(messagename) = LOWER('selectAction') AND LOWER(stateviewcurrent_pageelements_standardized_name) IN (LOWER('support-category.voice.reset-equip.contact-us'),LOWER('support-category.internet.reset-equip.contact-us'),LOWER('support-category.tv.reset-equip.contact-us')),1,0)) AS manual_reset_failures

    Page Views: Contact Us
        definition:
            SUM(IF(LOWER(visit_application_detailsapplication_name) = LOWER('SpecNet') AND LOWER(messagename) = LOWER('pageView') AND LOWER(stateviewcurrent_page_page_name) = LOWER('contactUs'),1,0)) AS contact_us_page_views

    Page Views Manual Troubleshooting:
        definition: SUM(IF(LOWER(visit_application_detailsapplication_name) = LOWER('SpecNet') AND LOWER(messagename) = LOWER('pageView') AND LOWER(stateviewcurrent_page_page_name) IN (LOWER('my-voice-services.manually-reset-equipment',LOWER('my-tv-services.manually-reset-equipment'),LOWER('my-internet-services.manually-reset-equipment')),1,0)) AS manual_troubleshoot_page_views

    MSA Support
        definition: visit__application_details__application_name = 'MySpectrum' AND message__name = 'pageViews' AND state__view__current_page__app_section = 'support'

    Button clicks add a line:
        definition: visit__application_details__application_name = 'SpecMobile' AND message__name = 'selectAction' AND state__view__current_page__elements__standardized_name = 'addLine'

    Late payments

    Account setup workflow completion

    Auto-pay enrollment rate

    Button clicks: Equip Troubleshoot Modal close





I don't know these (or we have not implemented them yet):
    Live contact rate
    Contact information: completeness
    Delinquent account rate
    Digital payment penetration rate
    Downgrade rate
    Load and send/receive time
    Monthly recurring revenue
    Number of clicks before task completion (avg.)
    Number of searches before engaging (avg)
    Paperless bill enrollment rate
    Refresh rate: TV box 'self-service'
    Successful 'Self-Service' internet reset rate
    Successful login attempts over total attempts
    Successful self-service to truck roll ratio
    Total payment cost
    Truck roll rate
    Upgrade conversion rate
    Utilization rate: Business email
    Utilization rate: Cloud backup
    Utilization rate: Consumer security suite
    Utilization rate: Domain
    Utilization rate: email
    Workflow completion rate: Authentication
    Workflow completion rate: Business email activation
    Workflow completion rate: Call scheduler setup
    Workflow completion rate: Cloud backup
    Workflow completion rate: Contact number update
    Workflow completion rate: Domain
    Workflow completion rate: Equipment
    Workflow completion rate: Paperless billing
    Workflow completion rate: Password reset
    Workflow completion rate: Preference management
    Workflow completion rate: Recover credentials
    Workflow completion rate: Review bill
    Workflow completion rate: Screening/blocking setup
    Workflow completion rate: Security suite
    Workflow completion rate: Self-service appointment
    Workflow completion rate: Sub-account creation/update
    Workflow completion rate: User management
    Workflow completion rate: Username creation
    Workflow completion rate: Username recovery
    Workflow completion rate: Voicemail setup
    Workflow completion rate: Web hosting
    Workflow completion rate: appointment management

















