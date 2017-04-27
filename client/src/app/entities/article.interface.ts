export interface IArticle {
    _id: string,
    created: Date,
    updated: Date,
    authors: [{keyname: string, forenames: string}],
    title: string,
    comments: string,
    'msc-class': string,
    'journal-ref': string,
    doi: string,
    license: string,
    abstract: string,
    arXiv_id: string,
    references: [any],
    fulltext_text: string
}