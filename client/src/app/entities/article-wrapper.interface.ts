import { IArticle } from './article.interface.js'

export interface IArticleWrapper {
    errno: number,
    articleJson: [IArticle]
}